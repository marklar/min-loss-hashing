function cb = mark_compactbit(b, wordsize)

% b: mtx

% cb = compacted string of bits
% (using words of 'wordsize' bits)

% default wordsize is 8
if (~exist('wordsize'))
  wordsize = 8;
end

if (wordsize == 8)
  type = 'uint8';
elseif (wordsize == 16)
  type = 'uint16';
elseif (wordsize == 32)
  type = 'uint32';
elseif (wordsize == 64)
  type = 'uint64';
else
  error('unrecognized wordsize');
end

% -- create mtx CB, all zeros --
%
% b: the code.
%   rows: bits
%   cols: samples
[nBits nSamples] = size(b);
% how many words do we need for out code?
nWords = ceil(nBits/wordsize);
% nWords-by-nSamples mtx of zeros of type 'type'.
cb = zeros([nWords nSamples], type);

% 
for bitNum = 1:nBits
  % w: the number of the byte we're in, starting at 1.
  w = ceil(bitNum/wordsize);
  % take current value in row: cd(w,:)
  % 
  cb(w,:) = bitset(
    cb(w,:),                    % array (row w)
    mod(bitNum-1,wordsize)+1,   % position in array
    b(bitNum,:)                 % val at row bitNum: {0,1}
  );
end
