    # def tile_process(self):
    #     """It will first crop input images to tiles, and then process each tile.
    #     Finally, all the processed tiles are merged into one images.
    #     Modified from: https://github.com/ata4/esrgan-launcher
    #     """
    #     batch, channel, height, width = self.img.shape
    #     output_height = height * self.scale
    #     output_width = width * self.scale
    #     output_shape = (batch, channel, output_height, output_width)
    #     tile_size=256
    #     tile_pad=32
    #     # start with black image
    #     self.output = self.img.new_zeros(output_shape)
    #     tiles_x = math.ceil(width / tile_size)
    #     tiles_y = math.ceil(height / tile_size)
    #
    #     # loop over all tiles
    #     for y in range(tiles_y):
    #         for x in range(tiles_x):
    #             # extract tile from input image
    #             ofs_x = x * tile_size
    #             ofs_y = y * tile_size
    #             # input tile area on total image
    #             input_start_x = ofs_x
    #             input_end_x = min(ofs_x + tile_size, width)
    #             input_start_y = ofs_y
    #             input_end_y = min(ofs_y + tile_size, height)
    #
    #             # input tile area on total image with padding
    #             input_start_x_pad = max(input_start_x - tile_pad, 0)
    #             input_end_x_pad = min(input_end_x + tile_pad, width)
    #             input_start_y_pad = max(input_start_y - tile_pad, 0)
    #             input_end_y_pad = min(input_end_y + tile_pad, height)
    #
    #             # input tile dimensions
    #             input_tile_width = input_end_x - input_start_x
    #             input_tile_height = input_end_y - input_start_y
    #             tile_idx = y * tiles_x + x + 1
    #             input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
    #
    #             # upscale tile
    #             try:
    #                 if hasattr(self, 'net_g_ema'):
    #                     self.net_g_ema.eval()
    #                     with torch.no_grad():
    #                         output_tile = self.net_g_ema(input_tile)
    #                 else:
    #                     self.net_g.eval()
    #                     with torch.no_grad():
    #                         output_tile = self.net_g(input_tile)
    #             except RuntimeError as error:
    #                 print('Error', error)
    #             print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')
    #
    #             # output tile area on total image
    #             output_start_x = input_start_x * self.opt['scale']
    #             output_end_x = input_end_x * self.opt['scale']
    #             output_start_y = input_start_y * self.opt['scale']
    #             output_end_y = input_end_y * self.opt['scale']
    #
    #             # output tile area without padding
    #             output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
    #             output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
    #             output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
    #             output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']
    #
    #             # put tile into output image
    #             self.output[:, :, output_start_y:output_end_y,
    #                         output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
    #                                                                    output_start_x_tile:output_end_x_tile]
    #
