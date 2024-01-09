from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import markdown2

import logging
from config import config, Args
from connection_manager import ConnectionManager, ServerFullException
import uuid
import time
from types import SimpleNamespace
from util import pil_to_frame, bytes_to_pil, get_pipeline_class
from device import device, torch_dtype
import asyncio
import os
import time
import torch
import gfx2cuda as g2c
import torchvision.transforms

THROTTLE = 1.0 / 120

torch.backends.cuda.matmul.allow_tf32 = True

class App:
	def __init__(self, config: Args, pipeline):
		self.args = config
		self.pipeline = pipeline
		self.app = FastAPI()
		self.conn_manager = ConnectionManager()
		self.init_app()

	def init_app(self):
		self.app.add_middleware(
			CORSMiddleware,
			allow_origins=["*"],
			allow_credentials=True,
			allow_methods=["*"],
			allow_headers=["*"],
		)

		@self.app.websocket("/api/ws/{user_id}")
		async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
			try:
				await self.conn_manager.connect(
					user_id, websocket, self.args.max_queue_size
				)
				await handle_websocket_data(user_id)
			except ServerFullException as e:
				logging.error(f"Server Full: {e}")
			finally:
				await self.conn_manager.disconnect(user_id)
				logging.info(f"User disconnected: {user_id}")

		async def handle_websocket_data(user_id: uuid.UUID):
			if not self.conn_manager.check_user(user_id):
				return HTTPException(status_code=404, detail="User not found")
			last_time = time.time()
			try:
				while True:
					if (
						self.args.timeout > 0
						and time.time() - last_time > self.args.timeout
					):
						await self.conn_manager.send_json(
							user_id,
							{
								"status": "timeout",
								"message": "Your session has ended",
							},
						)
						await self.conn_manager.disconnect(user_id)
						return
					data = await self.conn_manager.receive_json(user_id)
					if data["status"] == "next_frame":
						info = pipeline.Info()
						params = await self.conn_manager.receive_json(user_id)
						params = pipeline.InputParams(**params)
						params = SimpleNamespace(**params.dict())
						if info.input_mode == "image":
							image_data = await self.conn_manager.receive_bytes(user_id)
							if len(image_data) == 0:
								await self.conn_manager.send_json(
									user_id, {"status": "send_frame"}
								)
								continue
							params.image = bytes_to_pil(image_data)

						await self.conn_manager.update_data(user_id, params)
						await self.conn_manager.send_json(user_id, {"status": "wait"})

			except Exception as e:
				logging.error(f"Websocket Error: {e}, {user_id} ")
				await self.conn_manager.disconnect(user_id)

		@self.app.get("/api/queue")
		async def get_queue_size():
			queue_size = self.conn_manager.get_user_count()
			return JSONResponse({"queue_size": queue_size})

		@self.app.get("/api/stream/{user_id}")
		async def stream(user_id: uuid.UUID, request: Request):
			output_tensor = torch.ones((512, 512, 4), dtype=torch.uint8).mul(255).cuda()
			in_tens = torch.zeros((512, 512, 4), dtype=torch.uint8).cuda()
			output = g2c.texture(torch.ones((512, 512, 4), dtype=torch.uint8).cuda())
			g2c.set_global_texture(output)
			noise = torch.rand(3, 512, 512)
			noise_img  = torchvision.transforms.functional.to_pil_image(noise)
			frame = pil_to_frame(noise_img)
			handle = 2147501762
			input_texture = g2c.open_ipc_texture(handle)
			print(output.ipc_handle)
			try:
				async def generate():
					last_params = SimpleNamespace()

					while True:
						with input_texture as ptr: 
							input_texture.copy_to(in_tens)
							time.sleep(.016/4)

						last_time = time.time()
						await self.conn_manager.send_json(
							user_id, {"status": "send_frame"}
						)
						params = await self.conn_manager.get_latest_data(user_id)
						if params.__dict__ == last_params.__dict__ or params is None:
							await asyncio.sleep(THROTTLE)
							continue
						last_params = params

						without_alpha = in_tens[..., :3]

						for_img = without_alpha.permute(2, 0, 1).contiguous().mul(1/255).cuda()
						params.image = for_img
						params.width = 512
						params.height = 512

						pt_img = pipeline.predict(params)
						if pt_img is None:
							continue
						

						generated_tensor = pt_img.permute(1, 2, 0).contiguous().mul(255).cuda()
						
						output_tensor[..., :3] = generated_tensor

						g2c.texture(output_tensor)
						yield frame

						if self.args.debug:
							print(f"Time taken: {time.time() - last_time}")

				return StreamingResponse(
					generate(),
					media_type="multipart/x-mixed-replace;boundary=frame",
					headers={"Cache-Control": "no-cache"},
				)
			except Exception as e:
				logging.error(f"Streaming Error: {e}, {user_id} ")
				return HTTPException(status_code=404, detail="User not found")

		# route to setup frontend
		@self.app.get("/api/settings")
		async def settings():
			info_schema = pipeline.Info.schema()
			info = pipeline.Info()
			if info.page_content:
				page_content = markdown2.markdown(info.page_content)

			input_params = pipeline.InputParams.schema()
			return JSONResponse(
				{
					"info": info_schema,
					"input_params": input_params,
					"max_queue_size": self.args.max_queue_size,
					"page_content": page_content if info.page_content else "",
				}
			)

		if not os.path.exists("public"):
			os.makedirs("public")

		self.app.mount(
			"/", StaticFiles(directory="frontend/public", html=True), name="public"
		)


pipeline_class = get_pipeline_class(config.pipeline)
pipeline = pipeline_class(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
	import uvicorn

	uvicorn.run(
		"main:app",
		host=config.host,
		port=config.port,
		reload=config.reload,
		ssl_certfile=config.ssl_certfile,
		ssl_keyfile=config.ssl_keyfile,
	)
