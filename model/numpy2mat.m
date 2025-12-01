function mat = numpy2mat(file,Ns,Nc,Nx,Ny)
%% converts numpy array to MATLAB double array
    assert(isa(file,'py.numpy.ndarray')) % check file type
    mat = double(py.array.array('d', file.flatten())); % convert to 1D double (incorrect dimensions)
    shape = [Ny,Nx,Nc,Ns]; % dimensions are reversed compared to the documentation
    mat = reshape(mat, shape); % reshape to correct dimensions
    mat = permute(mat,[4 3 2 1]); % reverse dimensions to match documentation
end