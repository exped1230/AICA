%该函数利用冷暖色的membership function计算颜色的冷暖调
%输入值hue为颜色的色调,hue为0到1之间的值
%hue可以是一个值，也可以是一个向量
%输出值为该色调所对应的冷暖调的程度
function WarmAndCold=WarmCold(hue)

n=length(hue);
hue=hue*2*pi;
warm=zeros(n,1);
cold=zeros(n,1);
for i=1:n
    if hue(i)<=deg2rad(140)
        warm(i)=cos(hue(i)-deg2rad(50));
        cold(i)=0;
    elseif hue(i)<=deg2rad(320)
        warm(i)=0;
        cold(i)=cos(hue(i)-deg2rad(230));
    else
        warm(i)=cos(hue(i)-deg2rad(50));
        cold(i)=0;
    end
end
neutral=1-warm-cold;
WarmAndCold=cat(2,warm,neutral,cold);

end