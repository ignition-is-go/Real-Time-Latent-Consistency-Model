from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import markdown2

import logging
from config import config, Args
from connection_manager import ConnectionManager, ServerFullException
import uuid
import time
from types import SimpleNamespace
from texture_manager import TextureManager
from util import get_pipeline_class
from device import device, torch_dtype
import os
import time
import torch
import gfx2cuda as g2c

THROTTLE = 1.0 / 120

SIZE=512

torch.backends.cuda.matmul.allow_tf32 = True

class App:
	def __init__(self, config: Args, pipeline):
		self.args = config
		self.pipeline = pipeline
		self.conn_manager = ConnectionManager()
		async def handleUpdateCallback(user_id, handle):
			return await self.conn_manager.send_json(user_id, { 
				"status": "output_handle",
				"output_handle": handle
			})
		self.texture_manager = TextureManager(
			on_handle_change=handleUpdateCallback, 
			pipeline=pipeline
		)

		@asynccontextmanager
		async def lifespan(app: FastAPI):
			yield
			self.texture_manager.cancel_all()
			return

		self.app = FastAPI(lifespan=lifespan)
		self.init_app()

	def init_app(self):
		self.app.add_middleware(
			CORSMiddleware,
			allow_origins=["*"],
			allow_credentials=True,
			allow_methods=["*"],
			allow_headers=["*"],
		)

		@self.app.websocket("/api/ws/{user_id}" )
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
				self.texture_manager.cancel(user_id)
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
					print(data)

					if data["status"] == "source_info":
						params = {}
						params = pipeline.InputParams(**params)
						params = SimpleNamespace(**params.dict())
						params.prompt = data["prompt"]
						params.negative_prompt = data["negative_prompt"]
						params.steps = data["steps"]
						params.strength = data["strength"]
						params.guidance_scale = data["guidance_scale"]
						await self.texture_manager.update_info(
							user_id, 
							int(data["width"]), 
							int(data["height"]), 
							int(data["handle"]), 
							params
						)

			except Exception as e:
				logging.error(f"Websocket Error: {e}, {user_id} ")
				await self.conn_manager.disconnect(user_id)

		@self.app.get("/api/queue")
		async def get_queue_size():
			queue_size = self.conn_manager.get_user_count()
			return JSONResponse({"queue_size": queue_size})

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


print(f"Device: {device}")
print(f"torch_dtype: {torch_dtype}")
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
	
