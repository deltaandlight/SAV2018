N=64;
dat1=importdata('C:\Users\hasee\Desktop\2018年7~8月事务\沈洁老师的论文\program\SAV\mywork\BDF3\BDF3A\A_new_class_\example6_fig6\BinaStepAndEnergy');
A=dat1;
dtime=A(:,1);
energy=A(:,2);
time=dtime+0;
denergy=energy-0;
for i=1:N
    time(i)=sum(dtime(1:i));
end
for i=1:N-1
    denergy(i)=energy(i)-energy(i+1);
end
denergy(N)=0;

% [AX,H1,H2] = plotyy(time,denergy,time,dtime,'plot');
% set(AX(1),'XColor','k','YColor','b');
% set(AX(2),'XColor','k','YColor','r');
% HH1=get(AX(1),'Ylabel');
% set(HH1,'String','Left Y-axis');
% set(HH1,'color','b');
% HH2=get(AX(2),'Ylabel');
% set(HH2,'String','Right Y-axis');
% set(HH2,'color','r');
% set(H1,'LineStyle','-');
% set(H1,'color','b');
% set(H2,'LineStyle',':');
% set(H2,'color','r');
% %set(gca,'yscale','log')
% %set(gca,'yscale','log')
% %set(gca,'xscale','log')
% legend([H1,H2],{'y1 = 200*exp(-0.05*x).*sin(x)';'y2 = 0.8*exp(-0.5*x).*sin(10*x)'});
% xlabel('Zero to 20 musec.');
% title('Labeling plotyy');

plot(time,energy)