import gfx2cuda as g2c
import torch
import time

handle = 3221240514


t = g2c.open_ipc_texture(handle)
in_tens = torch.zeros((256, 256, 4), dtype=torch.uint8).cuda()



base = torch.ones((256, 256, 4), dtype=torch.uint8).cuda()
g = g2c.texture(base)
g2c.set_global_texture(g)
print(g.ipc_handle)

while True:
    with t as ptr: 
        t.copy_to(in_tens)
    g2c.texture(in_tens)
    time.sleep(.016/4)
    pass