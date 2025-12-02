function mat = numpy2mat(file,Nx,Ny,Nc,Ns)
%% converts numpy array to MATLAB double array
    assert(isa(file,'py.numpy.ndarray')) % check file type
    mat = double(py.array.array('d', file.flatten())); % convert to 1D double (incorrect dimensions)
    shape = [Ny,Nx,Nc,Ns]; % dimensions are reversed compared to the documentation
    mat = reshape(mat, shape); % reshape to correct dimensions
    mat = permute(mat,[2 1 3 4]); % reorder dimensions to Nx x Ny x Nc x Ns
end