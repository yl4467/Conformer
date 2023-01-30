import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional 


MAX_SEQ_LEN = 4096

def get_single_block_row_attention(block_id, to_start_block_id, to_end_block_id, num_rand_blocks,
                                   window_block_left=1, window_block_right=1, global_block_left=1,
                                   global_block_right=1):
  """For a single row block get random row attention.
  Args:
    block_id: int. block id of row.
    to_start_block_id: int. random attention coloum start id.
    to_end_block_id: int. random attention coloum end id.
    num_rand_blocks: int. number of random blocks to be selected.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.
  Returns:
    row containing the random attention vector of size num_rand_blocks.
  """

  # list of to_blocks from which to choose random attention
  to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
  #print(len(to_block_list))
  # permute the blocks
  perm_block = np.random.permutation(to_block_list)
  #print(perm_block)

  # illegal blocks for the current block id, using window
  illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

  # Add blocks at the start and at the end
  illegal_blocks.extend(list(range(global_block_left)))
  illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

  # The second from_block cannot choose random attention on second last to_block
  if block_id == 1:
    illegal_blocks.append(to_end_block_id-2)

  # The second last from_block cannot choose random attention on second to_block
  if block_id == to_end_block_id - 2:
    illegal_blocks.append(1)
  
  selected_random_blokcs = []
  for i in range(to_end_block_id - to_start_block_id):
    if len(selected_random_blokcs) == num_rand_blocks:
      break
    if perm_block[i] not in illegal_blocks:
      selected_random_blokcs.append(perm_block[i])
    

  #print(num_raand_blocks, len(selected_random_blokcs))
  return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(seq_length, block_size, num_heads, plan_from_length, plan_num_rand_blocks, 
                                      window_block_left=1, window_block_right=1, global_block_top=1,
                                      global_block_bottom=1, global_block_left=1, global_block_right=1):
    """Create adjacency list of random attention.
    Args:
    seq_length: int. length of sequence.
    block_size: int. size of block in sequence.
    num_heads: int. total number of heads.
    plan_from_length: list. plan from lenght where num_rand are choosen from.
    plan_num_rand_blocks: list. number of rand blocks within the plan.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_top: int. number of blocks at the top.
    global_block_bottom: int. number of blocks at the bottom.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.
    Returns:
    adjacency list of size num_head where each element is of size
    from_seq_length//from_block_size-2 by num_rand_blocks
    """
    num_blocks = seq_length//block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(seq_length)
    # Random Attention adjajency list
    rand_attn = [np.zeros((num_blocks,np.sum(plan_num_rand_blocks[:max_plan_idx+1])),
                        dtype=np.int32) for i in range(num_heads)]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx+1):
        rnd_r_cnt = 0
        if plan_idx > 0:
      # set the row for all from_blocks starting from 0 to
      # plan_block_length[plan_idx-1]
      # column indx start fromm plan_block_length[plan_idx-1] and ends at
      # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1]))
                for blk_rw_idx in range(global_block_top, plan_block_length[plan_idx-1]):
                    for h in range(num_heads):
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                             block_id=blk_rw_idx,
                             to_start_block_id=plan_block_length[plan_idx - 1],
                             to_end_block_id=plan_block_length[plan_idx],
                             num_rand_blocks=plan_num_rand_blocks[plan_idx],
                             window_block_left=window_block_left,
                             window_block_right=window_block_right,
                             global_block_left=global_block_left,
                             global_block_right=global_block_right)
        for pl_id in range(plan_idx):
            if plan_num_rand_blocks[pl_id] == 0:
                continue
            for blk_rw_idx in range(plan_block_length[plan_idx-1], plan_block_length[plan_idx]):
                rnd_r_cnt = 0
                to_start_block_id = 0
                if pl_id > 0:
                    rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                    to_start_block_id = plan_block_length[pl_id-1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id+1]))
                    for h in range(num_heads):
                                # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                                        block_id=blk_rw_idx,
                                        to_start_block_id=to_start_block_id,
                                        to_end_block_id=plan_block_length[pl_id],
                                        num_rand_blocks=plan_num_rand_blocks[pl_id],
                                        window_block_left=window_block_left,
                                        window_block_right=window_block_right,
                                        global_block_left=global_block_left,
                                        global_block_right=global_block_right)
            if plan_num_rand_blocks[plan_idx] == 0:
                continue
    # print("Start from here")
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx-1]
            to_start_block_id = plan_block_length[plan_idx-1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):
                #print("head", h, rnd_r_cnt, curr_r_cnt)
                rand_attn[h][blk_rw_idx,rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                         block_id=blk_rw_idx,
                         to_start_block_id=to_start_block_id,
                         to_end_block_id=plan_block_length[plan_idx],
                         num_rand_blocks=plan_num_rand_blocks[plan_idx],
                         window_block_left=window_block_left,
                         window_block_right=window_block_right,
                         global_block_left=global_block_left,
                         global_block_right=global_block_right)
    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks - global_block_bottom, :]
    #print(rand_attn.shape)
    return rand_attn

def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.
    Args:
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.
    num_rand_blocks: int. Number of random chunks per row.
    Returns:
    plan_from_length: ending location of from block
    plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2*num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2*num_rand_blocks + 5)*from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5)*from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks//2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks//2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)
    #print(plan_num_rand_blocks)
    return plan_from_length, plan_num_rand_blocks

def bigbird_block_rand_mask(from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1):
    """Create adjacency list of random attention.
    Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.
    Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
    """
    rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size-1):
        start = i-2
        end = i
        if i == 1:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end+1) == last:
                rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i-1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end+1:last])))[:r]
    return rand_attn

def full_bigbird_mask(from_seq_length, to_seq_length, from_block_size, to_block_size,rand_attn):
    """Calculate BigBird attention pattern as a full dense matrix.
    Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    rand_attn: adjajency matrix for random attention.
    Returns:
     attention mask matrix of shape [from_seq_length, to_seq_length]
    """
    attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
    for i in range(1, (MAX_SEQ_LEN // from_block_size) - 1):
        attn_mask[(i) * from_block_size:(i + 1) * from_block_size,(i - 1) * to_block_size:(i + 2) * to_block_size] = 1
        #print(rand_attn.shape)
        if(i<len(rand_attn) or i==len(rand_attn)):
            for j in rand_attn[i - 1, :]:
                attn_mask[i * from_block_size:(i + 1) * from_block_size,j * to_block_size:(j + 1) * to_block_size] = 1
    attn_mask[:from_block_size, :] = 1
    attn_mask[:, :to_block_size] = 1
    attn_mask[:, -to_block_size:] = 1
    attn_mask[-from_block_size:, :] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)

def create_rand_mask_from_inputs(from_blocked_mask, to_blocked_mask, rand_attn, num_attention_heads, num_rand_blocks, 
                                 from_seq_length,from_block_size):
    """Create 4D attention mask from a 3D tensor mask.
    Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.
    Returns:
    float Tensor of shape [batch_size, num_attention_heads,
                           from_seq_length//from_block_size-2,
                           from_block_size, num_rand_blocks*to_block_size].
    """
    num_windows = from_seq_length // from_block_size - 2
    
    rand_mask = torch.reshape(torch.gather(to_blocked_mask, 1, rand_attn),(-1, num_attention_heads, num_windows,num_rand_blocks * from_block_size))
    rand_mask = torch.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],rand_mask)
    return rand_mask

def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    """Create 4D attention mask from a 3D blocked tensor mask.
    Args:
    from_blocked_mask: 3D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: 3D Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
    """
    exp_blocked_to_pad = torch.concat((to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],to_blocked_mask[:, 3:-1]), 2)
    band_mask = torch.einsum("BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
    band_mask = torch.unsqueeze(band_mask, 1)
    return band_mask

def create_attention_mask_from_input_mask(from_mask, to_mask):
    """Create attention mask from a 2D tensor mask.
    Args:
    from_mask: float32 Tensor of shape [batch_size, from_seq_length].
    to_mask: float32 Tensor of shape [batch_size, to_seq_length].
    Returns:
    float32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
    """
    mask = torch.einsum("BF,BT->BFT", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = torch.unsqueeze(mask, 1)
    return mask 

def bigbird_block_sparse_attention(query_layer, key_layer, value_layer,rand_attn, num_attention_heads,
                                   size_per_head, num_rand_blocks, from_seq_length, to_seq_length,
                                   from_block_size, to_block_size):
    """BigBird attention sparse calculation using blocks in linear time.
    Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.
    A pure function with a long argument list to allow easy use outside our
    framework.
    Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    band_mask: float32 Tensor of shape [batch_size, 1,
      from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
      The values should be 1 or 0. The attention scores will effectively be
      set to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_mask: float32 Tensor of shape [batch_size, 1, from_seq_length, 1].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    to_mask: float32 Tensor of shape [batch_size, 1, 1, to_seq_length].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_blocked_mask: float32 Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
      Same as from_mask, just reshaped.
    to_blocked_mask: float32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
      Same as to_mask, just reshaped.
    rand_attn: int32 Tensor of shape [num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks] specifying which
      blocks to attend to for each from sequence block (except 2 global ones).
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    num_rand_blocks: int. Number of random chunks per row.
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
    """
    assert from_seq_length//from_block_size == to_seq_length//to_block_size
    activation = torch.nn.Softmax()
    # repeat for batch size
    batch_size = query_layer.shape[0]
    rand_attn = torch.unsqueeze(rand_attn, 0)
    #rand_attn = torch.unsqueeze(rand_attn, 3)
    rand_attn = rand_attn.repeat(batch_size,1,1,1).to(torch.int64).cuda()
    #rand_attn = rand_attn.reshape(-1)
    #rand_mask = create_rand_mask_from_inputs(
    #  from_blocked_mask, to_blocked_mask, rand_attn,
    #  num_attention_heads, num_rand_blocks,
    #  from_seq_length, from_block_size)
    #print(rand_attn.shape)
    # Define shorthands
    b = batch_size
    h = num_attention_heads
    r = num_rand_blocks
    d = size_per_head
    m = from_seq_length
    n = to_seq_length
    wm = from_block_size
    wn = to_block_size
    #rand_attn = rand_attn.repeat(batch_size,1,1, wn, d//r).to(torch.int64).cuda()
    #print(rand_attn.shape)
    blocked_query_matrix = torch.reshape(query_layer, (-1, h, m // wm, wm, d))
    blocked_key_matrix = torch.reshape(key_layer, (-1, h, n // wn, wn, d))
    blocked_value_matrix = torch.reshape(value_layer, (-1, h, n // wn, wn, d))
    #######
    blocked_key_matrix = blocked_key_matrix.cpu().detach().numpy()
    blocked_value_matrix = blocked_value_matrix.cpu().detach().numpy()
    rand_attn = rand_attn.cpu().detach().numpy()
    gathered_key = blocked_key_matrix[..., rand_attn[0,0,:,:], :, :]
    gathered_value = blocked_value_matrix[:, :,rand_attn[0,0,:,:], :, :]
    #print(gathered_key.shape)
    #####
    blocked_key_matrix = torch.tensor(blocked_key_matrix).cuda()
    blocked_value_matrix = torch.tensor(blocked_value_matrix).cuda()
    gathered_key = torch.reshape(torch.tensor(gathered_key),(-1, h, m // wm - 2, r * wn, d)).cuda()
    gathered_value = torch.reshape(torch.tensor(gathered_value),(-1, h, m // wm - 2, r * wn, d)).cuda()
    rand_attn = torch.tensor(rand_attn).cuda()
    #gathered_key = torch.reshape(torch.gather(blocked_key_matrix, 2, rand_attn),(-1, h, m // wm - 2, r * wn, d))
    #print(gathered_key.shape)
    #gathered_key = torch.reshape(torch.index_select(blocked_key_matrix, 2, rand_attn),(b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
    #gathered_value = torch.reshape(torch.gather(blocked_value_matrix,2,rand_attn),(-1, h, m // wm - 2, r * wn, d))
    #gathered_value = torch.reshape(torch.index_select(blocked_value_matrix, 2,rand_attn),(b, h, m // wm - 2, r * wn, -1))  
    # [b, h, n//wn-2, r, wn, -1]
    #print(gathered_key.shape)
    first_product = torch.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, 0], key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    first_product = torch.multiply(first_product, 1.0 / np.sqrt(d))
    #first_product += (1.0 - to_mask) * -10000.0
    first_attn_weights = activation(first_product)  # [b, h, wm, n]
    first_context_layer = torch.einsum("bhqk,bhkd->bhqd", first_attn_weights, value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    first_context_layer = torch.unsqueeze(first_context_layer, 2)

    second_key_mat = torch.cat([blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],blocked_key_matrix[:, :, 2],blocked_key_matrix[:, :, -1],gathered_key[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
    second_value_mat = torch.cat([blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],gathered_value[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
    second_product = torch.einsum("bhqd,bhkd->bhqk", blocked_query_matrix[:, :, 1], second_key_mat)  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    #second_seq_pad = torch.cat([to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
    #second_rand_pad = torch.cat([torch.ones_like(second_product[:, :, :, :4 * wn]), rand_mask[:, :, 0]], 3)
    second_product = torch.multiply(second_product, 1.0 / np.sqrt(d))
    #second_product += (1.0 - torch.minimum(second_seq_pad, second_rand_pad)) * -10000.0
    second_attn_weights = activation(second_product)  # [b , h, wm, (4+r)*wn]
    #print(second_attn_weights.shape, second_value_mat.shape)
    second_context_layer = torch.einsum("bhqk,bhkd->bhqd", second_attn_weights, second_value_mat)  
    # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_context_layer = torch.unsqueeze(second_context_layer, 2)

    exp_blocked_key_matrix = torch.cat([blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
      blocked_key_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
    exp_blocked_value_matrix = torch.cat([ 
      blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2], 
      blocked_value_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
    middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
    inner_band_product = torch.einsum(
      "bhlqd,bhlkd->bhlqk", middle_query_matrix, exp_blocked_key_matrix
    )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
    #     ==> [b, h, m//wm-4, wm, 3*wn]
    inner_band_product = torch.multiply(inner_band_product, 1.0 / np.sqrt(d))
    rand_band_product = torch.einsum(
      "bhlqd,bhlkd->bhlqk", middle_query_matrix, gathered_key[:, :, 1:-1]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
    #     ==> [b, h, m//wm-4, wm, r*wn]
    rand_band_product = torch.multiply(rand_band_product, 1.0 / np.sqrt(d))
    first_band_product = torch.einsum(
      "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    first_band_product = torch.multiply(first_band_product, 1.0 / np.sqrt(d))
    last_band_product = torch.einsum(
      "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
    last_band_product = torch.multiply(last_band_product, 1.0 / np.sqrt(d))
    #inner_band_product += (1.0 - band_mask) * -10000.0
    #first_band_product += (
    #  1.0 - torch.unsqueeze(to_mask[:, :, :, :wn], 3)) * -10000.0
    #last_band_product += (
    #  1.0 - torch.unsqueeze(to_mask[:, :, :, -wn:], 3)) * -10000.0
    #rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
    band_product = torch.cat([
      first_band_product, inner_band_product, rand_band_product,
      last_band_product], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
    attn_weights = activation(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn]
    context_layer = torch.einsum(
      "bhlqk,bhlkd->bhlqd", attn_weights[:, :, :, :, wn:4 * wn],
      exp_blocked_value_matrix
    )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
    #     ==> [b, h, m//wm-4, wm, -1]
    context_layer += torch.einsum(
      "bhlqk,bhlkd->bhlqd", attn_weights[:, :, :, :, 4 * wn:-wn],
      gathered_value[:, :, 1:-1]
    )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
    #     ==> [b, h, m//wm-4, wm, -1]
    context_layer += torch.einsum(
      "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, :wn],
      blocked_value_matrix[:, :, 0]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
    context_layer += torch.einsum(
      "bhlqk,bhkd->bhlqd", attn_weights[:, :, :, :, -wn:],
      blocked_value_matrix[:, :, -1]
    )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

    second_last_key_mat = torch.cat([
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
      blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
    second_last_value_mat = torch.cat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
      blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
    second_last_product = torch.einsum(
      "bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -2], second_last_key_mat
    )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
    #second_last_seq_pad = torch.cat([
    #  to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
    #  torch.ones_like(rand_mask[:, :1, 0, :1])], 3)
    #second_last_rand_pad = torch.cat(
    #  [torch.ones_like(second_last_product[:, :, :, :4 * wn]),
    #   rand_mask[:, :, -1]], 3)
    second_last_product = torch.multiply(second_last_product, 1.0 / np.sqrt(d))
    #second_last_product += (
    #  1.0 - tf.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
    second_last_attn_weights = activation(second_last_product)  # [b, h, wm, (4+r)*wn]
    second_last_context_layer = torch.einsum(
      "bhqk,bhkd->bhqd", second_last_attn_weights, second_last_value_mat
    )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
    second_last_context_layer = torch.unsqueeze(second_last_context_layer, 2)

    last_product = torch.einsum(
      "bhqd,bhkd->bhqk", blocked_query_matrix[:, :, -1], key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
    last_product = torch.multiply(last_product, 1.0 / np.sqrt(d))
    #last_product += (1.0 - to_mask) * -10000.0
    last_attn_weights = activation(last_product)  # [b, h, wm, n]
    last_context_layer = torch.einsum(
      "bhqk,bhkd->bhqd", last_attn_weights,
      value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
    last_context_layer = torch.unsqueeze(last_context_layer, 2)

    context_layer = torch.cat([
      first_context_layer, second_context_layer, context_layer,
      second_last_context_layer, last_context_layer
    ], 2)
    context_layer = torch.reshape(context_layer, (-1, h, m, d)) 
    context_layer = context_layer.permute(0, 2, 1, 3)
    return context_layer

class Dense3dLayer(nn.Module):
    """A dense layer with 3D kernel."""
    def __init__(self,
               num_attention_heads, size_per_head,
               name=None, head_first=False, use_bias=True):
        """Constructor for dense layer with 3D kernel.
        Args:
        num_attention_heads: The size of output dimension.
        size_per_head: The size per attention head.
        initializer: Kernel initializer.
        activation: Actication function.
        name: The name scope of this layer.
        head_first: Whether to output head dimension before or after sequence dim.
        use_bias: Whether the layer uses a bias vector.
        """
        super(Dense3dLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        #self.initializer = initializer
        self.head_first = head_first
        self.use_bias = use_bias

        hidden_size = self.num_attention_heads * self.size_per_head
        self.w = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(hidden_size, hidden_size),std =0.02), requires_grad=True)
        if self.use_bias:
            self.b = torch.nn.Parameter(torch.nn.init.constant_(torch.Tensor(hidden_size),0), requires_grad=True)     
        else:
            self.b = None

    def forward(self, input_tensor):
        """Constructor for dense layer with 3D kernel.
        Args:
        input_tensor: float Tensor of shape [batch, seq_length, hidden_size].
        Returns:
        float logits Tensor.
        """
        hidden_size = self.num_attention_heads * self.size_per_head
        reshape_w = torch.reshape(
            self.w, (hidden_size, self.num_attention_heads, self.size_per_head))
        if self.head_first:
            ret = torch.einsum("abc,cde->adbe", input_tensor, reshape_w)
        else:
            ret = torch.einsum("abc,cde->abde", input_tensor, reshape_w)
        if self.use_bias:
            if self.head_first:
                reshape_b = torch.reshape(self.b, (1, self.num_attention_heads, 1, self.size_per_head))
            else:
                reshape_b = torch.reshape(self.b, (self.num_attention_heads, self.size_per_head))
            ret += reshape_b
        return ret

class MultiHeadedAttentionLayer(nn.Module):
    """A multi-headed attention layer.
    It implements following types of multi-headed attention:
    - original_full attention from "Attention is all you Need".
    - simulated_sparse attention from BigBird with full quadratic implemention.
    - block_sparse attention from BigBird with memory efficient linear impl.
    """
    def __init__(self, attention_type, d_model = 512, num_attention_heads=8, size_per_head=64, num_rand_blocks=4, from_seq_length=96,
                to_seq_length=96, from_block_size=6, to_block_size=6, attention_probs_dropout_prob=0.2,
                initializer_range=0.02, use_bias=True, seed=None, query_act=None, key_act=None, value_act=None, name=None):
        """Constructor for a multi-headed attention layer.
        Args:
        attention_type: Type of attention, needs to be one of ['original_full',
        'simulated_sparse', 'block_sparse'].
        num_attention_heads: (optional) int. Number of attention heads.
        size_per_head: (optional) int. Size of each attention head.
        num_rand_blocks: (optional) int. Number of random chunks per row.
        from_seq_length: int. (optional) length of from sequence.
        to_seq_length: int. (optional) length of to sequence.
        from_block_size: (optional) int. size of block in from sequence.
        to_block_size: (optional) int. size of block in to sequence.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
        initializer_range: (optional) float. Range of the weight initializer.
        use_bias: Whether the layer uses a bias vector.
        seed: (Optional) int. Reandom seed for generating random mask.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        name: The name scope of this layer.
        """
        super(MultiHeadedAttentionLayer, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.num_rand_blocks = num_rand_blocks
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size
        self.seed = seed
        self.activation = torch.nn.Softmax()

        self.query_layer = Dense3dLayer(
          num_attention_heads, size_per_head,
          "query", head_first=True, use_bias=use_bias)

        self.key_layer = Dense3dLayer(
          num_attention_heads, size_per_head,
          "key", head_first=True, use_bias=use_bias)

        self.value_layer = Dense3dLayer(
           num_attention_heads, size_per_head,
          "value", head_first=True, use_bias=use_bias)
        #self.query_projection = nn.Linear(d_model, size_per_head*num_attention_heads)
        #self.key_projection = nn.Linear(d_model, size_per_head*num_attention_heads)
        #self.value_projection = nn.Linear(d_model, size_per_head*num_attention_heads)

        if attention_type == "original_full":
            self.attn_impl = self.original_full_attention
        elif attention_type == "simulated_sparse":
            self.attention_dropout = lambda x, training=None: x
            self.rand_attn = self.generate_rand_attn_list()
            #print(self.rand_attn.shape)
            self.rand_block_mask = self.convert_attn_list_to_mask(self.rand_attn)
            self.attn_impl = self.bigbird_simulated_attention
        elif attention_type == "block_sparse":
            assert from_seq_length//from_block_size == to_seq_length//to_block_size, (
            "Error the number of blocks needs to be same!")
            self.attention_dropout = None
            self.rand_attn = self.generate_rand_attn_list()
            self.attn_impl = self.bigbird_block_sparse_attention
        else:
            raise NotImplementedError(
            "Attention type {} is not implemented".format(attention_type))

    def generate_rand_attn_list(self):
    # generate random attention and corresponding masks
        if self.seed is not None:
            np.random.seed(self.seed)
    # old plans used in paper
        if self.from_seq_length in [1024, 2048, 3072, 4096]:
            rand_attn = [
              bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
              MAX_SEQ_LEN, MAX_SEQ_LEN,
              self.from_block_size, self.to_block_size, self.num_rand_blocks,
              last_idx=1024
              )[:(self.from_seq_length // self.from_block_size - 2)]
          for _ in range(self.num_attention_heads)
         ]
        else:
            plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
            self.from_seq_length, self.from_block_size, self.num_rand_blocks)
            #print(plan_from_length, plan_num_rand_blocks)
            rand_attn = bigbird_block_rand_mask_with_head(
               seq_length=self.from_seq_length,
               block_size=self.from_block_size,
               num_heads=self.num_attention_heads,
               plan_from_length=plan_from_length,
               plan_num_rand_blocks=plan_num_rand_blocks)
            rand_attn = np.stack(rand_attn, axis=0)
            #print(rand_attn)
        return torch.tensor(rand_attn, dtype=torch.int32)

    def convert_attn_list_to_mask(self, rand_attn):
        temp_mask = [
            full_bigbird_mask(  # pylint: disable=g-complex-comprehension
            self.from_seq_length, self.to_seq_length,
            self.from_block_size, self.to_block_size,
            rand_attn=rand_attn[m])
            for m in range(self.num_attention_heads)
        ]
        temp_mask = np.stack(temp_mask, axis=0)
        temp_mask = np.array(temp_mask, dtype=bool)
        rand_block_mask = torch.tensor(temp_mask, dtype=torch.bool)  # [N, F, T]
        #print(rand_block_mask.shape)
        return torch.tensor(rand_block_mask, dtype= torch.float32)

    def original_full_attention(self, query_layer, key_layer,value_layer, masks, training=None):
        """Full quadratic attention calculation.
        Args:
        query_layer: float Tensor of shape [batch_size, num_attention_heads,
        from_seq_length, size_per_head]
        key_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        value_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        masks: a list containing float32 Tensor representing attention_mask
        of shape [batch_size, from_seq_length, to_seq_length].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and
        will be unchanged for positions that are 1.
        training: Boolean indicating whether the call is training or inference.
        Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
        """
        attention_mask = masks

        # Directly take n^2 dot product between "query" and "key".
        attention_scores = torch.einsum("BNFH,BNTH->BNFT", query_layer, key_layer)
        attention_scores = torch.multiply(attention_scores, 1.0 / np.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - attention_mask) * -10000.0
            #print(adder.shape, attention_scores.shape)
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
            attention_scores.add_(adder.cuda())
    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
        
        attention_probs = self.activation(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
        #attention_probs = self.attention_dropout(attention_probs, training=training)

    # `context_layer` = [B, F, N, H]
        context_layer = torch.einsum("BNFT,BNTH->BFNH", attention_probs, value_layer)
        return context_layer

    def bigbird_simulated_attention(self, query_layer, key_layer, value_layer, masks, training=None):
        """BigBird attention calculation using masks in quadratic time.
        Args:
        query_layer: float Tensor of shape [batch_size, num_attention_heads,
        from_seq_length, size_per_head]
        key_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        value_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        masks: a list containing float32 Tensor representing attention_mask
        of shape [batch_size, from_seq_length, to_seq_length].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and
        will be unchanged for positions that are 1.
        training: Boolean indicating whether the call is training or inference.
        Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
        """
        attention_mask = masks
        rand_block_mask = torch.unsqueeze(self.rand_block_mask, 0)  # [1, N, F, T]
        if attention_mask is not None:
            attention_mask = torch.minimum(attention_mask, rand_block_mask)
        else:
            attention_mask = rand_block_mask
        return self.original_full_attention(query_layer, key_layer, value_layer, [attention_mask],training=training)

    def bigbird_block_sparse_attention(self, query_layer, key_layer, value_layer,
                                       masks, training=None):
        """BigBird attention sparse calculation using blocks in linear time.
        Args:
        query_layer: float Tensor of shape [batch_size, num_attention_heads,
        from_seq_length, size_per_head]
        key_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        value_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
        masks: A list of 5 masks used in BigBird attention at position 1 to 5.
        Position 0 (first element) is not used can be left as none. In the mask,
        the values should be 1 or 0. The attention scores will effectively
        be set to -infinity for any positions in the mask that are 0,
        and will be unchanged for positions that are 1.
           "None": Not needed.
            "band_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length//from_block_size-4,
              from_block_size, 3*to_block_size].
            "from_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length, 1].
            "to_mask": (optional) float32 Tensor of shape
              [batch_size, 1, 1, to_seq_length].
            "from_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length//from_block_size, from_block_size].
              Same as from_mask, just reshaped.
            "to_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, to_seq_length//to_block_size, to_block_size].
              Same as to_mask, just reshaped.}
        training: Boolean indicating whether the call is training or inference.
        Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
        """

        #(_, band_mask, from_mask, to_mask,
        #from_blocked_mask, to_blocked_mask) = masks

        return bigbird_block_sparse_attention(query_layer, key_layer, value_layer,self.rand_attn, 
        self.num_attention_heads, self.size_per_head,
        self.num_rand_blocks, self.from_seq_length, self.to_seq_length, 
        self.from_block_size, self.to_block_size)

    def forward(self, from_tensor, to_tensor,
           masks, training=None):
        """Implements a multi-headed attention layer from from_tensor to to_tensor.
        Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width]
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        masks: A list of masks used in different attention. Only relevant masks
        need to be supplied and at other positions place None. In the mask,
        the values should be 1 or 0. The attention scores will effectively
        be set to -infinity for any positions in the mask that are 0,
        and will be unchanged for positions that are 1.
           "attention_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length, to_seq_length].
            "band_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length//from_block_size-4,
              from_block_size, 3*to_block_size].
            "from_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length, 1].
            "to_mask": (optional) float32 Tensor of shape
              [batch_size, 1, 1, to_seq_length].
            "from_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length//from_block_size, from_block_size].
              Same as from_mask, just reshaped.
            "to_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, to_seq_length//to_block_size, to_block_size].
              Same as to_mask, just reshaped.}
        cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
        decode_i: (Used during prediction) current location of decoding
        training: Boolean indicating whether the call is training or inference.
        Returns:
        float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
        Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
        NotImplementedError: For unknown attention type.
        """

    # Scalar dimensions referenced here:
    #   b = batch size (number of sequences)
    #   m = `from_tensor` sequence length
    #   n = `to_tensor` sequence length
    #   h = `num_attention_heads`
    #   d = `size_per_head`
        
        #B, L, _ = from_tensor.shape
        #_, S, _ = to_tensor.shape
        #H = self.num_attention_heads

        #query = self.query_projection(from_tensor).view(B, H, L, -1)
        #key = self.key_projection(to_tensor).view(B, H, S, -1)
        #value = self.value_projection(to_tensor).view(B, H, S, -1)
        #print(query.shape)
    # `query` = [b, h, m, d]
        query = self.query_layer(from_tensor)
    # `key` = [b, h, n, d]
        key = self.key_layer(to_tensor)
    # `value_layer` = [b, h, n, d]
        value = self.value_layer(to_tensor)
        contextual_output = self.attn_impl(query, key, value, masks, training=training)
        #print(contextual_output.shape)
        return contextual_output.reshape(contextual_output.shape[0],contextual_output.shape[1],-1)

class birdattenalyer(nn.Module):
    def __init__(self, attention):
        super(birdattenalyer, self).__init__()
        self.attention = attention
    def forward(self,to_tensor, from_tensor):
        out = self.attention(to_tensor,from_tensor,masks=None)
        #print(out.shape)
        return out