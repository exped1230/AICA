%此函数对输入的图像进行分水岭分割，输出分割矩阵
function L=WaterFallSegment(rgb)
if ndims(rgb) == 3
    I = rgb2gray(rgb);
else
    I = rgb;
end

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);

%L = watershed(gradmag);
%Lrgb = label2rgb(L);
 
se = strel('square', 10);

Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);

Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);

fgm = imregionalmax(Iobrcbr);
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 20);
bw = im2bw(Iobrcbr, graythresh(Iobrcbr));

D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;

gradmag2 = imimposemin(gradmag, bgm | fgm4);

L = watershed(gradmag2);

% Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
% figure('units', 'normalized', 'position', [0 0 1 1]);
% subplot(1, 2, 1); imshow(rgb, []); title('原图像');
% subplot(1, 2, 2); imshow(Lrgb); title('彩色分水岭标记矩阵');
% 
% figure('units', 'normalized', 'position', [0 0 1 1]);
% subplot(1, 2, 1); imshow(rgb, []); title('原图像');
% subplot(1, 2, 2); imshow(rgb, []); hold on;
% himage = imshow(Lrgb);
% set(himage, 'AlphaData', 0.3);
% title('标记矩阵叠加到原图像');