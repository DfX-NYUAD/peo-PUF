

% 1. The Frequency (Monobit) Test,
% 2. Frequency Test within a Block,
% 3. The Runs Test,
% 4. Tests for the Longest-Run-of-Ones in a Block,
% 5. The Binary Matrix Rank Test,
% 6. The Discrete Fourier Transform (Spectral) Test,
% 7. The Non-overlapping Template Matching Test,
% 8. The Overlapping Template Matching Test,
% 9. Maurer's "Universal Statistical" Test,
% 10. The Linear Complexity Test,
% 11. The Serial Test,
% 12. The Approximate Entropy Test,
% 13. The Cumulative Sums (Cusums) Test,
% 14. The Random Excursions Test, and
% 15. The Random Excursions Variant Test. 
function [metrics,metrics_disp] = NIST_check(data,test)

if ~nargin
    data = '0011010101011011011110001000010010001011100100101110';
    test = 'all';
    Mb = 20;%5;
    [Mr,Qr] = deal(32);%Mr = 3; Qr = 3; % binary matrix
    m = 3; % overlapping,non-overlapping,serial,entropy
    Mo = 10; K = 5; % overlapping test
    Lm = 5; Qm = 3; % Maurier Universal 
elseif nargin == 1
    test = 'all';
    Mb = 20;%5;
    [Mr,Qr] = deal(32);%Mr = 3; Qr = 3; % binary matrix
    m = 3; % overlapping,non-overlapping,serial,entropy
    Mo = 10; K = 5; % overlapping test
    Lm = 5; Qm = 3; % Maurier Universal
else
    Mb = 20;%5;
    [Mr,Qr] = deal(32);%Mr = 3; Qr = 3; % binary matrix
    m = 3; % overlapping,non-overlapping,serial,entropy
    Mo = 10; K = 5; % overlapping test
    Lm = 5; Qm = 3; % Maurier Universal
end

global epsilon epsilon_bits
% epsilon = '0011010101011011011110001000010010001011100100101110';
% epsilon = data;
% epsilon_bits = double(epsilon)-48;
% n=length(epsilon);

% Testing
allTest = {'frequency(n)','blockFrequency(n,Mb)','runs(n)','longestRunOfOnes(n)',...
            'binaryMatrixRank(n,Mr,Qr)','discreteFourierTransform(n)',...
            'nonOverlappingTemplateMatching(n,m)',...
            'overlappingTemplateMatching(n,m,Mo,K)','serialTest(n,m)',... % Linear
            'maurerUniversal(n,Lm,Qm)','approximateEntropy(n,m)',...
            'cumulativeSums(n)','randomExcursions(n)',...
            'randomExcursionsVariant(n)'};
% allOperands = struct('n',n,'Mr',3,'Qr',3,'M',10,'K',5,'m',3);


if iscell(test) %list of test
elseif strcmpi(test,'all') %all test
    for ii=1:length(allTest)
        NumData = size(data,1);
        for jj=1:NumData
            epsilon = data(jj,:);
            epsilon_bits = double(epsilon)-48;
            n=length(epsilon);
            if strcmp(allTest{ii}(1:strfind(allTest{ii},'(')-1),'approximateEntropy')
                [temp1,ApEn,ActEn] = eval(allTest{ii});
                metrics.(allTest{ii}(1:strfind(allTest{ii},'(')-1))(jj,:) = ...
                temp1;
                metrics.ApEn(jj,:) = ApEn;
                metrics.ActEn(jj,:) = ActEn; % Actual Entropy
            else
                metrics.(allTest{ii}(1:strfind(allTest{ii},'(')-1))(jj,:) = ...
                    eval(allTest{ii});
            end
        end
    end
else %a single test
end

mFields = fieldnames(metrics);
num_mFields = length(mFields);
metrics_disp = struct();
pValue = 0.01;
for itr = 1:num_mFields
    mtest = mFields{itr};
    metrics_disp.(mtest){1} = metrics.(mtest);
    metrics_disp.(mtest){2} = sum(metrics.(mtest) >= pValue)/...
        length(metrics.(mtest)) * 100;
end

end
%% IGAMC function
function g = igamc(a,x)
g = gammainc(x,a,'upper');
end
%% 1. The Frequency (Monobit) Test, (<0.01 , non random)
function p_value = frequency(n)
global epsilon epsilon_bits
s = epsilon_bits; s(s==0)=-1;
s_obs = abs(sum(s))/sqrt(n);
p_value = erfc(s_obs);
end
%% 2. Frequency Test within a Block, (<0.01 , non random)
function p_value = blockFrequency(n,M)
global epsilon epsilon_bits
% M = 5;
N = floor(n/M);
PI_i = sum(reshape(epsilon_bits(1:N*M),M,N))/M;
chi2 = 4*M*sum((PI_i-1/2).^2);
p_value = igamc(N/2,chi2/2);
end
%% 3. The Runs Test,
function p_value = runs(n)
global epsilon epsilon_bits
pi_user = sum(epsilon_bits)/n;
tau = 2/sqrt(n);
t = abs(pi_user - 1/2);
if t >= tau
    p_value = 0;
else
    Vn_obs = sum(abs(epsilon_bits(2:end)-epsilon_bits(1:end-1))) + 1;
    p_value = erfc(abs(Vn_obs - 2*n*pi_user*(1-pi_user)) / (2*sqrt(2*n)*pi_user*(1-pi_user)) );
    if p_value < 0.01
    end
end
end
%% 4. Tests for the Longest-Run-of-Ones in a Block,
function p_value = longestRunOfOnes(n)
global epsilon epsilon_bits
nSet = [128,6272,750000];
mSet = [8,128,10^4];
vSet = {1:4,4:9,10:16};
NSet = [16,49,75];
PIset = {[0.21484375,0.3671875,0.23046875,0.1875]...
        ,[0.1174035788,0.242955959,0.249363483,0.17517706,0.102701071 ...
                ,0.112398847]...
        ,[0.0882,0.2092,0.2483,0.1933,0.1208,0.0675,0.0727]};
idx = find(n>=nSet, 1, 'last' );
if isempty(idx), idx=1; end
M = mSet(idx); V = vSet{idx};VN = NSet(idx);
N = floor(n/M);
[vClass] = deal(zeros(1,N));
for ii = 1:N
    temp = epsilon( ((ii-1)*M + 1):min(ii*M,n));
    temp1 = strsplit(strip(temp,'0'),'0');
    vClass(ii) = max(cellfun('length',temp1));
end
v=zeros(size(V));
VLen = length(V);

for ii=1:VLen
    if ii==1
        v(ii) = sum(vClass<=V(ii));
    elseif ii==VLen
        v(ii) = sum(vClass>=V(ii));
    else
        v(ii) = sum(vClass==V(ii));
    end
%     m = V(ii);
%     
%     pV = 0;
%     for r=0:M
%         U = min(M-r+1,floor(r/(m+1)));
%         p1 = 0;
%         for j = 0:U
%             p1 = p1 + ((-1)^j)*nchoosek(M-r+1,j)*nchoosek(M-j*(m+1),M-r);
%         end
%         p1 = p1/nchoosek(M,r);
%         pV = pV + (nchoosek(M,r)*p1)/2^M;
%     end
%     PIi(ii)= pV;
end
PIi = PIset{idx};
chi2 = sum( (v - VN*PIi).^2 ./ (VN * PIi));
p_value = igamc((VLen-1)/2,chi2/2);
end
%% 5. The Binary Matrix Rank Test,
function p_value = binaryMatrixRank(n,M,Q)
% M The number of rows in each matrix. For the test suite, 
%   M has been set to 32. If other values of M are used, new 
%   approximations need to be computed. 

% Q The number of columns in each matrix. For the test suite, Q has been
%   set to 32. If other values of Q are used, new approximations need 
%   to be computed.
global epsilon epsilon_bits
N = floor(n/(M*Q));
R = zeros(1,N);
for ii=1:N
    R(ii) = rank(reshape(epsilon_bits( (ii-1)*(M*Q)+1 : ii*(M*Q) ),Q,M)');
end
Fm = length(find(R==M));
Fm_1 = length(find(R==M));
m = min(M,Q);
R = 0:m;
p = zeros(1,3);
for ii = m-1:m+1
    r= R(ii);
    p1 = 2^(r*(Q+M-r)-M*Q);
    tQ = (0:r-1)-Q;
    tM = (0:r-1)-M;
    tr = (0:r-1)-r;
    p(ii-(m-2)) = p1 * prod(((1-2.^tQ).*(1-2.^tM))./(1-2.^tr));
end
F = [N-(Fm_1+Fm),Fm_1,Fm];
chi2 = sum(((F-2*p).^2) ./ (2*p));
p_value = igamc(1,chi2/2);
end
%% 6. The Discrete Fourier Transform (Spectral) Test,
function p_value = discreteFourierTransform(n)
global epsilon epsilon_bits
X = epsilon_bits;
S = fft(X);
M = abs(S(1:floor(n/2)));
T = sqrt(n*log(1/0.05));
N_0 = (0.95*n)/2;
N_1 = length(find(M<T));
d = (N_1 - N_0) / sqrt( (n*0.95*0.05)/4 );
p_value = erfc(abs(d)/sqrt(2));
end
%% 7. The Non-overlapping Template Matching Test,
function p_value = nonOverlappingTemplateMatching(n,m)
%m The length in bits of each template B. The template is the target string. 
%n The length of the entire bit string under test.
global epsilon epsilon_bits
numOfTemplates = [0, 2, 4, 6, 12, 20, 40, 74, 148, 284, 568, 1116,...
    2232, 4424, 8848, 17622, 35244, 70340, 140680, 281076, 562152];
Bset = cell(numOfTemplates(m),1);
fid = fopen(sprintf('templates/template%d',m));
for ii=1:length(Bset)
    temp = fgetl(fid);
    temp(temp==' ')=[];
    Bset{ii} = temp;
end
fclose(fid);
% m = length(B);
N = 8; M = floor(n/N);
if M < m
    N = min(100,n/(2*m));
    M = floor(n/M);
end
p_value = zeros(1,length(Bset));
for pp=1:length(p_value)
    W = zeros(1,N);
    B = Bset{pp};
    for ii = 1:N
        W(ii) = length(regexp(epsilon( (ii-1)*M+1 : ii*M ), B));
    end
    miu = (M-m+1)/2^m;
    sigma2 = M*(1/2^m - (2*m-1)/2^(2*m));
    chi2 = sum( (W-miu).^2 ) / sigma2;
    p_value(pp) = igamc(N/2,chi2/2);
end
end
%% 8. The Overlapping Template Matching Test,
function p_value = overlappingTemplateMatching(n,m,M,K)
%m length of the template B – in this case, the length of the run of ones. 
%n length of the entire bit string under test.
global epsilon epsilon_bits

if nargin < 3
    M = 1032;
    K = 5;
end
N = floor(n/M);
B = repmat('1',1,m);
v = zeros(1,6);
W = zeros(1,N);
for ii = 1:N
    W(ii) = length(strfind(epsilon( (ii-1)*M+1 : ii*M ), B));
end
lambda = (M-m+1)/2^m;
eta = lambda/2;
[v,PI_i] = deal(zeros(1,K+1));
for ii=0:K
    
    l = 1:ii;
    if isempty(l)
        A = 1;
        B = 1;
    else
        A = factorial(ii-1)./(factorial(ii-l).*factorial(l-1));
        B = eta.^l ./ factorial(l);
    end
%     PI_i(ii+1) = ( (eta*exp(-2*eta))/(2^ii) ) * hypergeom(ii+1,2,eta);
    PI_i(ii+1) = ( (exp(-eta))/(2^ii) ) * sum(A.*B);
    if ii==K
        v(ii+1) = length(find(W>=ii));
        PI_i(ii+1) = 1 - sum(PI_i(1:K));
    else
        v(ii+1) = length(find(W==ii));
    end
end
chi2 = sum( (v-N*PI_i).^2  ./ (N*PI_i) )  ;
p_value = igamc(N/2,chi2/2);
end
%% 9. Maurer's "Universal Statistical" Test,
function p_value = maurerUniversal(n,L,Q)
% L The length of each block. Note: the use of L as the block size is not 
%     consistent with the block size notation (M) used for the other tests.
%     However, the use of L as the block size was specified in the original
%     source of Maurer's test.
% Q The number of blocks in the initialization sequence.
% n The length of the bit string.
global epsilon epsilon_bits

K = floor(n/L) - Q;
bSet = 0:(2^L-1);
tSet = zeros(size(bSet));
sum = 0;
for ii=1:(K+Q)
    temp = bSet == bin2dec(epsilon( ((ii-1)*L + 1): ii*L )) ;
    old = tSet(temp);
    [new,tSet(temp)] = deal(ii);
    if ii > Q % testing
        sum = sum + log2(new - old);
    end
end

fn = sum / K;
% 
% N = L^5;
% A = (1-2^(-L)).^(0:N-1);
% B=log2(1:N);
% T=A.*B;
% Efn = 2^(-L) * sum(T); %expected value

EFN = [0.7326495, 1.5374383, 2.4016068, 3.3112247, 4.2534266, 5.2177052,...
       6.1962507, 7.1836656, 8.1764248, 9.1723243, 10.170032, 11.168765,...
       12.168070, 13.167693, 14.167488, 15.167379];
VAR = [0.690, 1.338, 1.901, 2.358, 2.705, 2.954, 3.125, 3.238, 3.311, ...
       3.356, 3.384, 3.401, 3.410, 3.416, 3.419, 3.421];
   
Efn = EFN(L); Var = VAR(L);
c = 0.7 - 0.8/L + (4 + 32/L)*(K^(-3/L) / 15);

sigma = c * sqrt(Var/K);

% p_value = erfc(abs( (fn - Efn)/(sqrt(2)*sigma) ));
p_value = erfc(abs( (fn - Efn)/(sqrt(2*Var)) ));
end
%% 10. The Linear Complexity Test,
% function p_value = linearComplexity(n,M)
% global epsilon epsilon_bits

%% 11. The Serial Test,
function p_value = serialTest(n,m)
% m The length in bits of each block.
% n The length in bits of the bit string.
global epsilon epsilon_bits
M = 2^m;

test_seq = [epsilon,epsilon(1:m-1)];
Vm = zeros(1,M);
for ii=1:M
    Vm(ii) = length(strfind(test_seq,dec2bin(ii-1,m)));
end
psi2 = ((2^m)/n)*sum((Vm - n/(2^m)).^2);

[psi2_1,psi2_2] = deal(0);
if m>1
    M1 = 2^(m-1);
    test_seq1 = [epsilon,epsilon(1:m-2)];
    Vm1 = zeros(1,M1);
    for ii=1:M1
        Vm1(ii) = length(strfind(test_seq1,dec2bin(ii-1,m-1)));
    end
    psi2_1 = ((2^(m-1))/n)*sum((Vm1 - n/(2^(m-1))).^2);
end

if m>2
    M2 = 2^(m-2);
    test_seq2 = [epsilon,epsilon(1:m-3)];
    Vm2 = zeros(1,M2);
    for ii=1:M2
        Vm2(ii) = length(strfind(test_seq2,dec2bin(ii-1,m-2)));
    end
    psi2_2 = ((2^(m-2))/n)*sum((Vm2 - n/(2^(m-2))).^2);
end
dPsi = psi2 - psi2_1;
d2Psi = psi2 - 2*psi2_1 + psi2_2;

p_value(1) = igamc(2^(m-2),dPsi/2);
p_value(2) = igamc(2^(m-3),d2Psi/2);
end
%% 12. The Approximate Entropy Test,
function [p_value,ApEn,ActEn] = approximateEntropy(n,m)
%m  The length of each block – in this case, the first block length used in
%   the test. m+1 is the second block length used.
%n  The length of the entire bit sequence.
global epsilon epsilon_bits
M = 2^m;
test_seq = [epsilon,epsilon(1:m-1)];
Cm = zeros(1,M);
for ii=1:M
    Cm(ii) = length(strfind(test_seq,dec2bin(ii-1,m)))/n;
end
M1 = 2^(m+1);
test_seq1 = [epsilon,epsilon(1:m)];
Cm1 = zeros(1,M1);
for ii=1:M1
    Cm1(ii) = length(strfind(test_seq1,dec2bin(ii-1,m+1)))/n;
end
Cm(Cm==0)=1;Cm1(Cm1==0)=1; % modify for accuracy
phi = sum(Cm .* log(Cm));
phi1 = sum(Cm1 .* log(Cm1));
ApEn = (phi - phi1);
p0 = sum(epsilon_bits == 0)/n;
p1 = 1 - p0;
ActEn = -(p0*log2(p0) + p1*log2(p1));
chi2 = 2*n*(log(2) - ApEn );
p_value = igamc(2^(m-1),chi2/2);
end
%% 13. The Cumulative Sums (Cusums) Test,
function p_value = cumulativeSums(n,mode)
if nargin == 1
    mode = 0;%0-forward,1-backward
end
global epsilon epsilon_bits
X = 2*epsilon_bits - 1;
switch mode
    case 0
        S = cumsum(X);
    case 1
        S = cumsum(fliplr(X));
end
z = max(abs(S));
k = floor( (-n/z + 1)/4 ) : floor( (n/z - 1)/4 );
sum1 = sum(normcdf(((4*k+1)*z) / sqrt(n)) - normcdf(((4*k-1)*z)/sqrt(n)));

k = floor( (-n/z - 3)/4 ) : floor( (n/z - 1)/4 );
sum2 = sum(normcdf(((4*k+3)*z) / sqrt(n)) - normcdf(((4*k+1)*z)/sqrt(n)));

p_value = 1 - sum1 + sum2;
end
%% 14. The Random Excursions Test, and
function p_value = randomExcursions(n)
global epsilon epsilon_bits
X = 2*epsilon_bits - 1;
S = cumsum(X);
S_ = [0,S,0];
Jp = find(S_ == 0); %Jp(Jp == 1) = [];
J = length(Jp) - 1;
V = zeros(8,6);
x = (-4:4)'; x(x==0) = [];
circ = cell(1,J);
for ii=1:J
    circle = S_(Jp(ii):Jp(ii+1));
    circ{ii} = circle;
    a = unique(circle(circle ~= 0));
    b = ~ismember(x,a);
    for jj=1:length(x)
        if b(jj) % not available
            V(jj,1) = V(jj,1) + 1;
        else % avilable
            vTemp = sum(circle(circle ~= 0) == x(jj));
            V(jj,1+min(5,vTemp)) = V(jj,1+min(5,vTemp)) + 1;
        end
    end
end
pi_k =  zeros(8,6);
pi_k(:,1) = 1 - 1./(2*abs(x));
for k = 1:4
    pi_k(:,k+1) = (1./(4*x.^2)) .* (1 - 1./(2*abs(x))).^(k-1);
end
pi_k(:,end) = (1./(2*abs(x))) .* (1 - 1./(2*abs(x))).^(4);

chi2 = sum( ( (V - J*pi_k).^2 ./ (J*pi_k) ) , 2);
p_value = igamc(5/2,chi2/2);
end
%% 15. The Random Excursions Variant Test. 
function p_value = randomExcursionsVariant(n)
global epsilon epsilon_bits
X = 2*epsilon_bits - 1;
S = cumsum(X);
S_ = [0,S,0];
Jp = find(S_ == 0); %Jp(Jp == 1) = [];
J = length(Jp) - 1;
x = (-9:9)'; x(x==0) = [];
eta = zeros(size(x));
for jj=1:length(x)
    eta(jj) = sum(S_ == x(jj));
end

p_value = erfc( abs(eta - J) ./ sqrt(2*J*(4*abs(x) - 2)) );
end