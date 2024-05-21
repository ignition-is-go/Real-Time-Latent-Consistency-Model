from types import SimpleNamespace
from uuid import UUID 
import torch
import gfx2cuda as g2c 
import asyncio
import time

class TextureTransfer:
	def __init__(self, on_handle_change: lambda handle: None, pipeline: any):
		self.source_handle = None
		self.width = None
		self.height = None
		self.on_handle_change = on_handle_change
		self.pipeline = pipeline
		self.params = SimpleNamespace()
		self.loopTask: asyncio.Future = None
		self.cancelEvent: asyncio.Event = None

	def cancel(self):
		if self.loopTask and self.cancelEvent:
			self.loopTask.cancel()
			self.cancelEvent.set()


	def setInputTexture(self, source_handle: int):
		self.source_handle = source_handle
		self.input_texture = g2c.open_ipc_texture(self.source_handle)
		return
	
	async def setProps(self, width: int, height: int, source_handle: int):
		reload = False
		if width != self.width or height != self.height:
			await self.setSize(width, height)
			reload = True
		if source_handle != self.source_handle:
			self.setInputTexture(source_handle)
			reload = True

		if reload:
			self.cancel()
			await self.run()
		return
	
	async def setSize(self, width: int, height: int):
		self.width =  width
		self.height = height
		self.output_tensor = torch.ones((self.height, self.width, 4), dtype=torch.uint8).mul(255).cuda()
		self.input_tensor = torch.zeros((self.height, self.width, 4), dtype=torch.uint8).cuda()
		self.output_texture = g2c.texture(torch.ones((self.height, self.width, 4), dtype=torch.uint8).cuda())
		self.output_handle = self.output_texture.ipc_handle
		await self.on_handle_change(self.output_texture.ipc_handle)
		return

	def loop(self):
		while True:

			if self.cancelEvent.is_set():
				return

			with self.input_texture: 
					self.input_texture.copy_to(self.input_tensor)
		
			without_alpha = self.input_tensor[..., :3]
			
			for_img = without_alpha.permute(2, 0, 1).contiguous().mul(1/255).cuda()
			self.params.image = for_img
			self.params.width = self.width
			self.params.height = self.height
			pt_img = self.pipeline.predict(self.params)
			if pt_img is None:
				return

			generated_tensor = pt_img.permute(1, 2, 0).contiguous().mul(255).cuda()
			
			self.output_tensor[..., :3] = generated_tensor

			with self.output_texture:
				self.output_texture.copy_from(self.output_tensor)

			time.sleep(1/60)
	
	async def run(self):
		self.cancelEvent = asyncio.Event()
		self.loopTask = asyncio.get_event_loop().run_in_executor(None, lambda:  self.loop())

class TextureManager:
	def __init__(self, on_handle_change: lambda  user_id, handle: None, pipeline):
		self.textures = {}
		self.transfers: dict[UUID, TextureTransfer] = {}
		self.on_handle_change = on_handle_change
		self.pipeline = pipeline
		return

	def cancel(self, user_id: UUID):
		if user_id in self.transfers:
			self.transfers[user_id].cancel()
			del self.transfers[user_id]
		return
	
	def cancel_all(self):
		for user_id in self.transfers:
			self.cancel(user_id)
		return	
  
	async def update_info(self, user_id: UUID, width: int, height: int, handle: int, params: SimpleNamespace):
		if not user_id in self.transfers:
			async def cb(handle):
				await self.on_handle_change(user_id, handle)
				return
			self.transfers[user_id] = TextureTransfer(lambda handle: cb(handle), self.pipeline)
			self.transfers[user_id].params = params
			await self.transfers[user_id].setProps(width, height, handle)
		

		else:
			self.transfers[user_id].params = params
			await self.transfers[user_id].setProps(width, height, handle)
		return