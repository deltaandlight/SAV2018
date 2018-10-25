clear all;
L=128;
%L=256;
x=0:(1/(L-1)):1;
y=0:(1/(L-1)):1;
hx=2*pi/L;
hy=2*pi/L;

filepath='C:\Users\hasee\Desktop\2018年7~8月事务\沈洁老师的论文\program\SAV\mywork\BDF3\BDF3A\A_new_class_\example6_fig6\data\';
for i=0:64
%     if i<10
%         filename=strcat('ASCIsol_000',num2str(i));
%     else
%         filename=strcat('ASCIsol_00',num2str(i));
%     end
    if i<10
        filename=strcat(strcat('ASCIsol_00',num2str(i)),'0');
    else
        filename=strcat(strcat('ASCIsol_0',num2str(i)),'0');
    end
    dat1=importdata(strcat(filepath,filename),' ',4);
    A=reshape(dat1.data,L,L);
%     for j =1:L
%         for k =1:L
%             if mod(j+k,2)==1
%                 A(j,k)=-A(j,k);
%             end
%         end
%     end
    %A=dat1.data;
    %image(A)
    %colorbar
    %contour(A)
     %gcf=surf(A)
     %saveas(gcf,['myfig_surf',num2str(i),'.jpg'])
     %saveas(gcf,['myfig_contour',num2str(i),'.jpg'])
     image(A,'CDataMapping','scaled')
     saveas(gcf,['myfig_image',num2str(i),'.jpg'])
% hold on 
% image(A,'CDataMapping','scaled')

end
    

