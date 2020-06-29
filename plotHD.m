% HDdata = HD5;
% comment = 'Si 5um 0nm - Si 5um 60nm';

function [xd,pd] = plotHD(HDdata)%,comment)
mu = mean(HDdata);
sigma = std(HDdata);
% hammingDist(1).comment = comment;
% hammingDist(1).mean = mu;

h = histogram(HDdata,'Visible','off','Normalization','probability');%,'NumBins',25);%round(sqrt(length(HDdata))));
xh = deal((h.BinEdges(1:end-1)+h.BinEdges(2:end))/2);
ph = h.Values;
xd = xh;%linspace(h.BinEdges(1),h.BinEdges(end),2*h.NumBins-1);
% close(gcf)

gm = makedist('Normal','mu',mu,'sigma',sigma);
yd = pdf(gm,xd); pd = yd/sum(yd);

% hammingDist(1).x = xd;
% hammingDist(1).pd = pd;


% figure
bar(xh,ph)
hold on
plot(xd,pd,'LineWidth',1)
xlabel('Hamming Distance')
ylabel('Probability')
set(gca,'PlotBoxAspectRatio', [2 1 1])
xlim([0 0.8])
hold off
% title('HD')
% text(0.7*max(xd),0.7*max(pd),['\mu = ',num2str(mu)])
% text(0.7*max(xd),0.6*max(pd),['\sigma = ',num2str(sigma)])
% text(0.7*max(xd),0.4*max(pd),comment)
