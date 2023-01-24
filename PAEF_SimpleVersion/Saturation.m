%该函数利用饱和度的membership function计算饱和程度
%输入值color为颜色的饱和度,color为0到1之间的值
%color可以是一个值，也可以是一个向量
%输出值为该饱和度所对应的低、中、高饱和度的程度
function IttenSaturation=Saturation(color)

n=length(color);
color=color*100;
ls=zeros(n,1);
ms=zeros(n,1);
hs=zeros(n,1);

for i=1:n
    if color(i)<=10
        ls(i)=1;
        ms(i)=0;
        hs(i)=0;
    elseif color(i)<=27
        ls(i)=(27-color(i))/17;
        ms(i)=(color(i)-10)/17;
        hs(i)=0;
    elseif color(i)<=51
        ls(i)=0;
        ms(i)=(51-color(i))/24;
        hs(i)=(color(i)-27)/24;
    else
        ls(i)=0;
        ms(i)=0;
        hs(i)=1;
    end
end

IttenSaturation=cat(2,ls,ms,hs);

end