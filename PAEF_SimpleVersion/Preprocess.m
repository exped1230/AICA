%�ú���������ͼ�����Ԥ����
function [rgb,H,S,V,L]=Preprocess(Img)

%����IAPS�����ݿ���Ҫ�Ǻڱߵ��������ʹ��FrameCropping1��������
%�����������ɫ����򣬾͵�ʹ��FrameCropping��
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