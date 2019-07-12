function features = waveshape(data)

data = data(:);

w = 100e-3;
d = 0.06;
N = length(data);

FilterNN = zeros(N);

n = 1:N;
features = zeros(1,N);
for m=1:N
    a = (1-m)*d;
    
    wsFreqOffset = ( a + (n-1)*d)';
    FilterNN(:,m) = sech(2*log(sqrt(2)+1)*(wsFreqOffset/w));
%     features(m) = mean(FilterNN(:,m) .* data);
    features(m) = round(mean(FilterNN(:,m) .* data)/1e3);
end

