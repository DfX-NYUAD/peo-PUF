function key = getKey(data,len,spacing,ignoreEndLength)
%data : wavelength data to get keys from
%len : [6]length of bits per instance
%spacing : {[1],2}
%ignoreEnd :{[3],2,1,0} 

if ~exist('len','var')
    len = 6;
end
if ~exist('spacing','var')
    spacing = 1;
end
if ~exist('ignoreEndLength','var')
    ignoreEndLength = 3;
end

gCode = dec2bin(bitxor(data,bitshift(data,-1)));
d = len;
key = gCode(:,end-ignoreEndLength:-spacing:end-ignoreEndLength-d+1)'; key = key(:)';
end
