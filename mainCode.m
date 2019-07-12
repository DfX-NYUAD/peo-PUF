clear
close all
clc
%% Sheet 1
fprintf('Extracting data\n')
data.sheet1 = xlsread('photonic data\wavelength_time_dependent_transmission.xlsx','Sheet1');
names = {'no Au','r=60nm','r=120nm','r=180nm','r=240nm'};
comments = {'Si - r=5um, h=180nm';'Au - h=30nm'};
timeData = struct();
yt = double('BDJLT')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    timeData(ii).time = tempTime(~isnan(tempTime));
    timeData(ii).data = tempData(~isnan(tempData));
    timeData(ii).name = names{ii};
    plot(timeData(ii).time,timeData(ii).data)
    hold on
end
hold off
xlabel('time (ps)')
ylabel('transmission (arb.)')
legend(names)

wavelengthData = struct();
xt = double('EGMOQ')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    wavelengthData(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData(ii).data = tempData(~isnan(tempData));
    wavelengthData(ii).name = names{ii};
    plot(wavelengthData(ii).lambda,wavelengthData(ii).data)
    hold on
end
hold off
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')
legend(names)
%% Sheet 2
fprintf('Extracting data\n')
data.sheet2 = xlsread('photonic data\wavelength_time_dependent_transmission.xlsx','Sheet2');
names = {'no Au','r=60nm','r=120nm','r=240nm'};
comments = {'Si - r=6um, h=180nm';'Au - h=30nm'};
timeData2 = struct();
yt = double('JLNP')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = data.sheet2(:,xt(ii));
    tempData = data.sheet2(:,yt(ii));
    timeData2(ii).time = tempTime(~isnan(tempTime));
    timeData2(ii).data = tempData(~isnan(tempData));
    timeData2(ii).name = names{ii};
    plot(timeData2(ii).time,timeData2(ii).data)
    hold on
end
hold off
xlabel('time (ps)')
ylabel('transmission (arb.)')
legend(names)

wavelengthData2 = struct();
xt = double('ACEG')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = data.sheet2(:,xt(ii));
    tempData = data.sheet2(:,yt(ii));
    wavelengthData2(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData2(ii).data = tempData(~isnan(tempData));
    wavelengthData2(ii).name = names{ii};
    plot(wavelengthData2(ii).lambda,wavelengthData2(ii).data)
    hold on
end
hold off
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')
legend(names)
%% Au updated
fprintf('Extracting data\n')
data.sheet1 = xlsread('photonic data\data collected_19.11.2018.csv');
names = {'1 Au nanodisk','5 Au nanodisk'};

timeData3 = struct();
yt = double('DH')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    timeData3(ii).time = tempTime(~isnan(tempTime));
    timeData3(ii).data = tempData(~isnan(tempData));
    timeData3(ii).name = names{ii};
    plot(timeData3(ii).time,timeData3(ii).data)
    hold on
end
hold off
legend(names);
xlabel('time (ps)')
ylabel('transmission (arb.)')

wavelengthData3 = struct();
xt = double('AE')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    wavelengthData3(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData3(ii).data = tempData(~isnan(tempData));
    wavelengthData3(ii).name = names{ii};
    plot(wavelengthData3(ii).lambda,wavelengthData3(ii).data)
    hold on
end
hold off
legend(names)
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')

%% 1 Au nanodisk with different pulsewidth
fprintf('Extracting data\n')
data.sheet1 = xlsread('photonic data\data collected_19.11.2018_different_pulsewidth_1_Au_nanoparticle.csv');
names = {'pulsewidth 50fs','pulsewidth 100fs','pulsewidth 200fs'};
comment = '1 Au nanodisk';

timeData4 = struct();
yt = double('HJL')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    timeData4(ii).time = tempTime(~isnan(tempTime));
    timeData4(ii).data = tempData(~isnan(tempData));
    timeData4(ii).name = names{ii};
    plot(timeData4(ii).time,timeData4(ii).data)
    hold on
end
hold off
legend(names);
xlabel('time (ps)')
ylabel('transmission (arb.)')

wavelengthData4 = struct();
xt = double('ACE')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    wavelengthData4(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData4(ii).data = tempData(~isnan(tempData));
    wavelengthData4(ii).name = names{ii};
    plot(wavelengthData4(ii).lambda,wavelengthData4(ii).data)
    hold on
end
hold off
legend(names)
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')


%% 5 nanodisk with different impurities
fprintf('Extracting data\n')
data.sheet1 = xlsread('photonic data\data collected_21.11.2018.csv');
names = {'Au','TiN'};
comment = 'Si Linear 5 nanodisks';

timeData5 = struct();
yt = double('FH')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    timeData5(ii).time = tempTime(~isnan(tempTime));
    timeData5(ii).data = tempData(~isnan(tempData));
    timeData5(ii).name = names{ii};
    plot(timeData5(ii).time,timeData5(ii).data)
    hold on
end
hold off
legend(names);
xlabel('time (ps)')
ylabel('transmission (arb.)')

wavelengthData5 = struct();
xt = double('AC')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = data.sheet1(:,xt(ii));
    tempData = data.sheet1(:,yt(ii));
    wavelengthData5(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData5(ii).data = tempData(~isnan(tempData));
    wavelengthData5(ii).name = names{ii};
    plot(wavelengthData5(ii).lambda,wavelengthData5(ii).data)
    hold on
end
hold off
legend(names)
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')

%% Temperature based data
fname = 'photonic data\Au_r=60nm_x=3um_y=2um_wavelength_t=8ps_T=300K_dn_to_dt=0.00018.txt';
fname1 = 'photonic data\Au_r=60nm_x=3um_y=2um_wavelength_t=8ps_T=350K_dn_to_dt=0.00018.txt';

data = csvread(fname,3,0); data=data(end:-1:1,:);
data1 = csvread(fname1,3,0); data1=data1(end:-1:1,:);

c300_350 = conv(data(:,2),data1(:,2));
[~,p300_350] = max(c300_350);
p300_350 = size(data,1) - rem(p300_350,100);

data_new = data(1:end-p300_350-1,:);
data1_new = data1(p300_350+2:end,:); 
data1_new(:,1) = data1(1:end-p300_350-1,1);
namesT = {'300K Au 60nm Si 5um','350K Au 60nm Si 5um'};

temperatureData(1).data = data_new(:,2);
temperatureData(2).data = data1_new(:,2);

figure
plot(data_new(:,1),data_new(:,2))
hold on
plot(data1_new(:,1),data1_new(:,2))
hold off
legend(namesT)
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')

%% 5 nanodisk with different impurities
fprintf('Extracting data\n')
mydata.sheet1 = xlsread('photonic data\data collected_25.11.2018_1.csv');
names = {'Au linear','TiN linear','TiN nonlinear'};
comment = 'Si Linear/nonlinear 5 nanodisks';

timeData6 = struct();
yt = double('HJL')-double('A')+1;
xt = yt - 1;
figure
for ii=1:length(xt)
    tempTime = mydata.sheet1(:,xt(ii));
    tempData = mydata.sheet1(:,yt(ii));
    timeData6(ii).time = tempTime(~isnan(tempTime));
    timeData6(ii).data = tempData(~isnan(tempData));
    timeData6(ii).name = names{ii};
    plot(timeData6(ii).time,timeData6(ii).data)
    hold on
end
hold off
legend(names);
xlabel('time (ps)')
ylabel('transmission (arb.)')

wavelengthData6 = struct();
xt = double('ACE')-double('A')+1;
yt = xt + 1;
figure
for ii=1:length(xt)
    tempLambda = mydata.sheet1(:,xt(ii));
    tempData = mydata.sheet1(:,yt(ii));
    wavelengthData6(ii).lambda = tempLambda(~isnan(tempLambda));
    wavelengthData6(ii).data = tempData(~isnan(tempData));
    wavelengthData6(ii).name = names{ii};
    plot(wavelengthData6(ii).lambda,wavelengthData6(ii).data)
    hold on
end
hold off
legend(names)
xlabel('wavelength (nm)')
ylabel('transmission (arb.)')

%% Getting the keys
fprintf('Getting the binaries\n')
len = 3;
spacing = 1;
ignorePart = 1;
for ii=1:length(wavelengthData)
    dataIn = waveshape(wavelengthData(ii).data);
    wavelengthData(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end

for ii=1:length(wavelengthData2)
    dataIn = waveshape(wavelengthData2(ii).data);
    wavelengthData2(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end

for ii=1:length(wavelengthData3)
    dataIn = waveshape(wavelengthData3(ii).data);
    wavelengthData3(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end

for ii=1:length(wavelengthData4)
    dataIn = waveshape(wavelengthData4(ii).data);
    wavelengthData4(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end

for ii=1:length(wavelengthData5)
    dataIn = waveshape(wavelengthData5(ii).data);
    wavelengthData5(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end

temperatureData(1).binary = getKey(waveshape(temperatureData(1).data),len,spacing,ignorePart);
temperatureData(2).binary = getKey(waveshape(temperatureData(2).data),len,spacing,ignorePart);

for ii=1:length(wavelengthData6)
    dataIn = waveshape(wavelengthData6(ii).data);
    wavelengthData6(ii).binary = getKey(dataIn,len,spacing,ignorePart);
end
%% Grouping for similarity measurement
wavelength1 = cell2mat({wavelengthData.binary}');
names1 = {wavelengthData.name};

wavelength2 = cell2mat({wavelengthData2.binary}');
names2 = {wavelengthData2.name};

wavelength3 = cell2mat({wavelengthData3.binary}');
names3 = {wavelengthData3.name};

wavelength4 = cell2mat({wavelengthData4.binary}');
names4 = {wavelengthData4.name};

wavelength5 = cell2mat({wavelengthData5.binary}');
names5 = {wavelengthData5.name};

wavelengthT = cell2mat({temperatureData.binary}');
namesT = {'300K Au 60nm Si 5um','350K Au 60nm Si 5um'};

wavelength6 = cell2mat({wavelengthData6.binary}');
names6 = {wavelengthData6.name};

%getting the length of instance to use for proper keying{128 bits}
grp = ceil(128/len);

num = size(wavelength1,2)/len;
numT = size(wavelengthT,2)/len; % for shifted reduced temperature data

genTime = 1e3;

temp5um_1 = reshape(wavelength1(strcmp(names1', 'no Au'),:),len,[]);
temp5um_2 = reshape(wavelength1(strcmp(names1', 'r=60nm'),:),len,[]);
temp6um_1 = reshape(wavelength2(strcmp(names2', 'r=60nm'),:),len,[]);
temp6um_2 = reshape(wavelength2(strcmp(names2', 'r=240nm'),:),len,[]);

temp60_1 = reshape(wavelength1(strcmp(names1', 'r=60nm'),:),len,[]);
temp60_2 = reshape(wavelength2(strcmp(names2', 'r=60nm'),:),len,[]);
temp120_1 = reshape(wavelength1(strcmp(names1', 'r=120nm'),:),len,[]);
temp120_2 = reshape(wavelength2(strcmp(names2', 'r=120nm'),:),len,[]);
temp240_1 = reshape(wavelength1(strcmp(names1', 'r=240nm'),:),len,[]);
temp240_2 = reshape(wavelength2(strcmp(names2', 'r=240nm'),:),len,[]);
tempAu_1 = reshape(wavelength3(1,:),len,[]);
tempAu_2 = reshape(wavelength3(2,:),len,[]);

tempPulse_1 = reshape(wavelength4(1,:),len,[]);
tempPulse_2 = reshape(wavelength4(2,:),len,[]);
tempPulse_3 = reshape(wavelength4(3,:),len,[]);

tempMaterial_1 = reshape(wavelength5(1,:),len,[]); %Au
tempMaterial_2 = reshape(wavelength5(2,:),len,[]); %TiN

tempT_1 = reshape(wavelengthT(1,:),len,[]);
tempT_2 = reshape(wavelengthT(2,:),len,[]);

temp5Nano_1 = reshape(wavelength6(1,:),len,[]);%Au  / Linear Si /equal Size
temp5Nano_2 = reshape(wavelength6(2,:),len,[]);%TiN / Linear Si /equal Size
temp5Nano_3 = reshape(wavelength6(3,:),len,[]);%TiN / Linear Si /diff Size


[wavelength5um,wavelength6um,wavelength60,...
    wavelength120,wavelength240,wavelengthAu,...
    wavelengthP12,wavelengthP13,wavelengthP23,...
    wavelengthMat,wavelength_T,wavelength5Nano12,...
    wavelength5Nano13,wavelength5Nano23] = ...
    deal(cell(2,genTime));

[HD5,HD6,HDr60,HDr120,HDr240,HD_Au,...
    HD_Pulse12,HD_Pulse13,HD_Pulse23,HD_Mat,HDr60_Temperature,...
    HD_5Nano12,HD_5Nano13,HD_5Nano23] = ...
    deal(zeros(1,genTime));
fprintf('Computing HD\n')
for ii=1:genTime
    pos = randperm(num,grp); % random positions
    posT = randperm(numT,grp); % random positions (temperature data)
    
    %extracts the binaries
    wavelength5um{1,ii} = reshape(temp5um_1(:,pos),1,[]);
    wavelength5um{2,ii} = reshape(temp5um_2(:,pos),1,[]);
    wavelength6um{1,ii} = reshape(temp6um_1(:,pos),1,[]);
    wavelength6um{2,ii} = reshape(temp6um_2(:,pos),1,[]);
    wavelength60{1,ii} = reshape(temp60_1(:,pos),1,[]);
    wavelength60{2,ii} = reshape(temp60_2(:,pos),1,[]);
    wavelength120{1,ii} = reshape(temp120_1(:,pos),1,[]);
    wavelength120{2,ii} = reshape(temp120_2(:,pos),1,[]);
    wavelength240{1,ii} = reshape(temp240_1(:,pos),1,[]);
    wavelength240{2,ii} = reshape(temp240_2(:,pos),1,[]);
    wavelengthAu{1,ii} = reshape(tempAu_1(:,pos),1,[]);
    wavelengthAu{2,ii} = reshape(tempAu_2(:,pos),1,[]);
    
    wavelengthP12{1,ii} = reshape(tempPulse_1(:,pos),1,[]);
    wavelengthP12{2,ii} = reshape(tempPulse_2(:,pos),1,[]);
    wavelengthP13{1,ii} = reshape(tempPulse_1(:,pos),1,[]);
    wavelengthP13{2,ii} = reshape(tempPulse_3(:,pos),1,[]);
    wavelengthP23{1,ii} = reshape(tempPulse_2(:,pos),1,[]);
    wavelengthP23{2,ii} = reshape(tempPulse_3(:,pos),1,[]);
    
    wavelengthMat{1,ii} = reshape(tempMaterial_1(:,pos),1,[]);
    wavelengthMat{2,ii} = reshape(tempMaterial_2(:,pos),1,[]);
    
    wavelength_T{1,ii} = reshape(tempT_1(:,posT),1,[]);
    wavelength_T{2,ii} = reshape(tempT_2(:,posT),1,[]);
    
    wavelength5Nano12{1,ii} = reshape(temp5Nano_1(:,pos),1,[]);
    wavelength5Nano12{2,ii} = reshape(temp5Nano_2(:,pos),1,[]);
    wavelength5Nano13{1,ii} = reshape(temp5Nano_1(:,pos),1,[]);
    wavelength5Nano13{2,ii} = reshape(temp5Nano_3(:,pos),1,[]);
    wavelength5Nano23{1,ii} = reshape(temp5Nano_2(:,pos),1,[]);
    wavelength5Nano23{2,ii} = reshape(temp5Nano_3(:,pos),1,[]);
    
    %compute HD and randomness
    HD5(ii) = pdist(double(cell2mat(wavelength5um(:,ii)))-double('0'),'Hamming');
    HD6(ii) = pdist(double(cell2mat(wavelength6um(:,ii)))-double('0'),'Hamming');
    HDr60(ii) = pdist(double(cell2mat(wavelength60(:,ii)))-double('0'),'Hamming');
    HDr120(ii) = pdist(double(cell2mat(wavelength120(:,ii)))-double('0'),'Hamming');
    HDr240(ii) = pdist(double(cell2mat(wavelength240(:,ii)))-double('0'),'Hamming');
    HD_Au(ii) = pdist(double(cell2mat(wavelengthAu(:,ii)))-double('0'),'Hamming');
    
    HD_Pulse12(ii) = pdist(double(cell2mat(wavelengthP12(:,ii)))-double('0'),'Hamming');
    HD_Pulse13(ii) = pdist(double(cell2mat(wavelengthP13(:,ii)))-double('0'),'Hamming');
    HD_Pulse23(ii) = pdist(double(cell2mat(wavelengthP23(:,ii)))-double('0'),'Hamming');
    
    HD_Mat(ii) = pdist(double(cell2mat(wavelengthMat(:,ii)))-double('0'),'Hamming');
    HDr60_Temperature(ii) = pdist(double(cell2mat(wavelength_T(:,ii)))-double('0'),'Hamming');
    
    HD_5Nano12(ii) = pdist(double(cell2mat(wavelength5Nano12(:,ii)))-double('0'),'Hamming');
    HD_5Nano13(ii) = pdist(double(cell2mat(wavelength5Nano13(:,ii)))-double('0'),'Hamming');
    HD_5Nano23(ii) = pdist(double(cell2mat(wavelength5Nano23(:,ii)))-double('0'),'Hamming');
end
%% Reporting HD
HDdata = HD6;
comment = {'Si 6um 60nm','Si 6um 240nm'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(2).comment = comment;
hammingDist(2).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')


HDdata = HDr60;
comment = {'Si 5um 60nm','Si 6um 60nm'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(3).comment = comment;
hammingDist(3).mean = mu;

HDdata = HDr120;
comment = {'Si 5um 120nm','Si 6um 120nm'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(4).comment = comment;
hammingDist(4).mean = mu;

HDdata = HDr240;
comment = {'Si 5um 60nm','Si 6um 60nm'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(5).comment = comment;
hammingDist(5).mean = mu;

HDdata = HD_Au;
comment = {'1 Au','5 Au'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(6).comment = comment;
hammingDist(6).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_particles','svg')

HDdata = HD_Pulse12;
comment = {'pulsewidth 50fs','pulsewidth 100fs'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(7).comment = comment;
hammingDist(7).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')

HDdata = HD_Pulse13;
comment = {'pulsewidth 50fs','pulsewidth 200fs'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(8).comment = comment;
hammingDist(8).mean = mu;
gcf;hold on;%figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\intraHD_pulse','svg')

HDdata = HD_Pulse23;
comment = {'pulsewidth 100fs','pulsewidth 200fs'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(9).comment = comment;
hammingDist(9).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')

HDdata = HD_Mat;
comment = {'Material Au','Material TiN'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(10).comment = comment;
hammingDist(10).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_material','svg')

HDdata = HD5;
comment = {'Si 5um 0nm','Si 5um 60nm'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(1).comment = comment;
hammingDist(1).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')

HDdata = HDr60_Temperature;
comment = {'Temperature 300K','Temperature 350K'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(11).comment = comment;
hammingDist(11).mean = mu;
gcf;hold on%figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(0.01,0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(0.01,0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(0.01,0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_Temperature','svg')

HDdata = HD_5Nano12;
comment = {'5 Au linear','5 TiN linear'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(12).comment = comment;
hammingDist(12).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_5Au_linear_5TiN_linear','svg')

HDdata = HD_5Nano13;
comment = {'5 Au linear','5 TiN nonlinear'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(13).comment = comment;
hammingDist(13).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_5Au_linear_5TiN_nonlinear','svg')


HDdata = HD_5Nano23;
comment = {'5 TiN linear','5 TiN nonlinear'};
mu = mean(HDdata);sigma = std(HDdata);
hammingDist(14).comment = comment;
hammingDist(14).mean = mu;
figure('Name',strjoin(comment));
[xd,pd] = plotHD(HDdata);
title('HD')
text(max(xd),0.7*max(pd),['\mu = ',num2str(mu)],'color','red')
text(max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)],'color','red')
text(max(xd),0.4*max(pd),comment,'color','red')
saveas(gcf,'results\interHD_5TiN_nonlinear_5TiN_nonlinear','svg')

hammingDistTable = struct2table(hammingDist);
disp(hammingDistTable)


%% Compute randomness
fprintf('Estimating randomness\n')
[~,random5_1] = NIST_check(cell2mat(wavelength5um(1,:)'));
[~,random5_2] = NIST_check(cell2mat(wavelength5um(2,:)'));

[~,random6_1] = NIST_check(cell2mat(wavelength6um(1,:)'));
[~,random6_2] = NIST_check(cell2mat(wavelength6um(2,:)'));

[~,random60_1] = NIST_check(cell2mat(wavelength60(1,:)'));
[~,random60_2] = NIST_check(cell2mat(wavelength60(2,:)'));

[~,random120_1] = NIST_check(cell2mat(wavelength120(1,:)'));
[~,random120_2] = NIST_check(cell2mat(wavelength120(2,:)'));

[~,random240_1] = NIST_check(cell2mat(wavelength240(1,:)'));
[~,random240_2] = NIST_check(cell2mat(wavelength240(2,:)'));

[~,randomAu_1] = NIST_check(cell2mat(wavelengthAu(1,:)'));
[~,randomAu_2] = NIST_check(cell2mat(wavelengthAu(2,:)'));

[~,randomPulse_1] = NIST_check(cell2mat(wavelengthP12(1,:)'));
[~,randomPulse_2] = NIST_check(cell2mat(wavelengthP12(2,:)'));
[~,randomPulse_3] = NIST_check(cell2mat(wavelengthP13(2,:)'));

[~,randomMat_1] = NIST_check(cell2mat(wavelengthMat(1,:)'));
[~,randomMat_2] = NIST_check(cell2mat(wavelengthMat(2,:)'));

[~,randomTemp_1] = NIST_check(cell2mat(wavelength_T(1,:)'));
[~,randomTemp_2] = NIST_check(cell2mat(wavelength_T(2,:)'));

[~,random5Nano_1] = NIST_check(cell2mat(wavelength5Nano12(1,:)'));
[~,random5Nano_2] = NIST_check(cell2mat(wavelength5Nano12(2,:)'));
[~,random5Nano_3] = NIST_check(cell2mat(wavelength5Nano13(2,:)'));

fprintf('Reporting randomness\n')
randomTable(1,:) = struct2table(random60_1); 
rowName{1,:} = 'Si5umAu60nm';
randomTable(2,:) = struct2table(random60_2); 
rowName{2,:} = 'Si6umAu60nm';
randomTable(3,:) = struct2table(random120_1); 
rowName{3,:} = 'Si5umAu120nm';
randomTable(4,:) = struct2table(random120_2); 
rowName{4,:} = 'Si6umAu120nm';
randomTable(5,:) = struct2table(random240_1); 
rowName{5,:} = 'Si5umAu240nm';
randomTable(6,:) = struct2table(random240_2); 
rowName{6,:} = 'Si6umAu240nm';
randomTable(7,:) = struct2table(randomAu_1); 
randomTable(8,:) = struct2table(randomAu_2); 
rowName(7:8,:) = names3(:);
randomTable(9,:) = struct2table(random5_1); 
rowName{9,:} = 'Si5umAu0nm';

randomTable(10,:) = struct2table(randomPulse_1); 
rowName{10,:} = 'PulseWidth50fs';
randomTable(11,:) = struct2table(randomPulse_2); 
rowName{11,:} = 'PulseWidth100fs';
randomTable(12,:) = struct2table(randomPulse_3); 
rowName{12,:} = 'PulseWidth200fs';

randomTable(13,:) = struct2table(randomMat_1); 
rowName{13,:} = 'Material Au';
randomTable(14,:) = struct2table(randomMat_2); 
rowName{14,:} = 'Material TiN';

randomTable(15,:) = struct2table(randomTemp_1); 
rowName{15,:} = 'Temperature 300K';
randomTable(16,:) = struct2table(randomTemp_2); 
rowName{16,:} = 'Temperature 350K';

randomTable(17,:) = struct2table(random5Nano_1); 
rowName{17,:} = '5 Au linear';
randomTable(18,:) = struct2table(random5Nano_2); 
rowName{18,:} = '5 TiN linear';
randomTable(19,:) = struct2table(random5Nano_3); 
rowName{19,:} = '5 TiN nonlinear';


randomTable.Properties.RowNames = rowName;
disp(randomTable)

fprintf('Reporting randomness statistics\n')
variableNames = randomTable.Properties.VariableNames;

randomStatStruct = struct();

for ii=1:length(variableNames)
    randomStatStruct.(variableNames{ii}) = ...
        randomTable{:,variableNames{ii}}(:,2);
end


randomStatTable = struct2table(randomStatStruct);
randomStatTable.Properties.RowNames = rowName;
disp(randomStatTable)

writetable(randomStatTable,'results\PUF NIST.xlsx','WriteRowNames',true)

minEnt = cellfun(@min,randomTable.ActEn); minEnt = minEnt(:,1);
meanEnt = cellfun(@mean,randomTable.ActEn);meanEnt = meanEnt(:,1);
entropyTable = table(minEnt,meanEnt);
entropyTable.Properties.RowNames = rowName;
disp(entropyTable)

writetable(entropyTable,'results\PUF NIST.xlsx','Range','G27:I46','WriteRowNames',true)