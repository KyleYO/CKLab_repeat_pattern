img=imread('C:\Users\李佳謙\Desktop\論文進度\分水嶺\圖庫\re_1.jpg');
imshow(img);  
[m n]=size(img);  
img=double(img);  
%%canny????的前?步相?不复?，所以我就直接?用系?函?了  
%%高斯?波  
w=fspecial('gaussian',[5 5]);  
img=imfilter(img,w,'replicate');  
figure;  
imshow(uint8(img))  
%%sobel????  
w=fspecial('sobel');  
img_w=imfilter(img,w,'replicate');      %求???  
w=w';  
img_h=imfilter(img,w,'replicate');      %求???  
img=sqrt(img_w.^2+img_h.^2);        %注意?里不是??的求平均，而是平方和在?方。我曾?好?一段??都搞?了  
figure;  
imshow(uint8(img))  
% %%下面是非极大抑制  
% new_edge=zeros(m,n);  
% for i=2:m-1  
%     for j=2:n-1  
%         Mx=img_w(i,j);  
%         My=img_h(i,j);  
%           
%         if My~=0  
%             o=atan(Mx/My);      %??的法?弧度  
%         elseif My==0 && Mx>0  
%             o=pi/2;  
%         else  
%             o=-pi/2;              
%         end  
%           
%         %Mx?用My和img?行插值  
%         adds=pioter_coords(o);%??像素法?一?求得的??坐?，插值需要         
%         M1=My*img(i+adds(2),j+adds(1))+(Mx-My)*img(i+adds(4),j+adds(3));   %插值后得到的像素，用此像素和?前像素比?   
%         adds=pioter_coords(o+pi);%??法?另一?求得的??坐?，插值需要  
%         M2=My*img(i+adds(2),j+adds(1))+(Mx-My)*img(i+adds(4),j+adds(3));%另一?插值得到的像素，同?和?前像素比?  
%           
%         isbigger=(Mx*img(i,j)>M1)*(Mx*img(i,j)>=M2)+(Mx*img(i,j)<M1)*(Mx*img(i,j)<=M2); %如果?前?比???都大置1  
%           
%         if isbigger  
%            new_edge(i,j)=img(i,j);   
%         end          
%     end  
% end  
% figure;  
% imshow(uint8(new_edge))  
% %%下面是?后?值?理  
% up=120;     %上?值  
% low=100;    %下?值  
% set(0,'RecursionLimit',10000);  %?置最大??深度  
% for i=1:m  
%     for j=1:n  
%       if new_edge(i,j)>up &&new_edge(i,j)~=255  %判?上?值  
%             new_edge(i,j)=255;  
%             new_edge=pioter_connect(new_edge,i,j,low);  
%       end  
%     end  
% end  
% figure;  
% imshow(new_edge==255)  

