%该函数对输入图像进行预处理
function [rgb,H,S,V,L]=Preprocess(Img)

%由于IAPS等数据库主要是黑边的相框，所以使用FrameCropping1方法即可
%如果有其他颜色的相框，就得使用FrameCropping了
%Cropped=FrameCropping1(Img);
ImgChannel = length(size(Img));
if ImgChannel == 2
    Img = cat(3,Img,Img,Img);
elseif ImgChannel == 4
    Img = Img(:,:,1:3);
end
rgb=ImageResize(Img);
[H,S,V]=rgb2hsy(rgb);
L=WaterFallSegment(rgb);

% figure;imshow(Img);
% figure;imshow(rgb);
% figure;imshow(H);
% figure;imshow(S);
% figure;imshow(V);
% figure;imshow(L);

end