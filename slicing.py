import torch
            
def get_line_errors(out_full, out_patch, full_ids, patch_ids, threshold=1e-6):

    errors = []
    for (line_f, line_p) in zip([*range(full_ids[0],full_ids[1])], [*range(patch_ids[0],patch_ids[1])]):
        out_p_line = out_patch[0,:,line_p,:]
        out_line = out_full[0,:,line_f,:]
        errors.append(torch.abs(out_line - out_p_line).max())
        # if torch.isclose(out_line, out_p_line, atol=threshold).all():
        #     print(f"LineP {line_p} and line {line_f} is close - relative error: {torch.abs(out_line - out_p_line).max()}")
        # else:
        #     print(f"LineP {line_p} and line {line_f} is different - relative error: {torch.abs(out_line - out_p_line).max()}")
    
    return errors        
  
def learn_slice_flip(forward_fn, num_slices=4, slice_id=3, threshold=1e-6, input_shape=(1, 3, 160, 160)):
    
    input_tensor = torch.rand(input_shape)  
    output_tensor = forward_fn(input_tensor)
    out_h = output_tensor.shape[2]
    input_h = input_tensor.shape[2]
    in_stride, out_stride = max(1, input_h//out_h), max(1, out_h//input_h)
    print(f"Input size: {input_h}, Output size: {out_h} -> Input stride: {in_stride}, output stride: {out_stride}")
    
    if out_h % num_slices != 0:
        raise ValueError("The number of slices must be a divisor of the height of the output tensor")
    
    offset = out_h//num_slices * slice_id
    target_lines = [0, out_h//num_slices]
    
    start_row = input_h//num_slices * slice_id
    end_row =  input_h//num_slices * (slice_id+1)
    
    while True:
        
        print(f"Start row: {start_row}, end row: {end_row}")
        input_tensor_p = input_tensor[:,:,start_row:end_row,:]
        output_tensor_p = forward_fn(input_tensor_p)
        
        errors = get_line_errors(output_tensor, output_tensor_p, [offset, offset + out_h//num_slices], target_lines, threshold=threshold)
        e_first, e_last = errors[0], errors[-1]
        
        if e_last > threshold:
            print(f"Error {e_last:.4f} > thr, Updating end row")
            end_row += in_stride
            if end_row > input_tensor.shape[2]:
                end_row = input_tensor.shape[2]
                print(f"WARNING: End slice is too large - ERROR: {e_last:.4f}")
                e_last = -1
            print(f"End row: {end_row}")
        
        if e_first > threshold:
            print(f"Error {e_first:.4f} > thr, Updating start row")
            start_row -= in_stride
            target_lines[0] += out_stride
            target_lines[1] += out_stride
            if start_row < 0:
                start_row = 0
                print(f"WARNING: Start slice is too small - ERROR: {e_first:.4f}")
                e_first = -1
            print(f"Start row: {start_row}")
            
        if e_first <= threshold and e_last <= threshold:
            print(f"Start row: {start_row}, end row: {end_row} -> Target lines: {target_lines}")
            print(f"MSE: {torch.nn.functional.mse_loss(output_tensor_p[:,:,target_lines[0]:target_lines[1],:], output_tensor[:,:,offset:offset+out_h//num_slices,:])}")
            print(f"Max error: {torch.abs(output_tensor_p[:,:,target_lines[0]:target_lines[1],:] - output_tensor[:,:,offset:offset+out_h//num_slices,:]).max()}")
            return (start_row, end_row), target_lines
            
