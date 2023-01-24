Img = imread('test1.jpg');
[rgb,H,S,V,L]=Preprocess(Img);
tic
colornames = ColorNames(rgb);
colornames = colornames/sum(colornames);
IttenContrast = IttenColorContrast(H,S,V,L,colornames);
Principles = cat(2,colornames,IttenContrast);
toc