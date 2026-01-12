
# 1 Box_setting: 1 initial
## from Query Pos
where is query pos and what it is?
samples: NestedTensor, tensor: batched images, of shape [batch_size x 3 x H x W]
 - dense_input_embed = self.pos_embed(samples) # from RGB images, B * hidden_dim * imgH * imgW, self.pos_embed = position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
 - query_embed = dense_input_embed[:, :, points_queries[:, 0], points_queries[:, 1]] # pq is generated from fixed steps, serving as grids
 - query_pos = query_embed # now: input sending to the transformer
 - query_pos_2d = self.init_offset_generator(query_pos.reshape(-1, self.d_model)) # self.init_offset_generator = MLP(d_model, d_model, 2, 3), initial the offset from query pos
 - 