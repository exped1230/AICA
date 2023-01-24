%该函数计算Contrast of Extension
%输入colornames为各种颜色的像素个数
function Contrast=ContrastOfExtension(colornames)

Contrast=std(colornames);