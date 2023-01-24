%该函数利用饱和度的membership function计算光照程度
%输入值value为颜色的亮度
%value可以是一个值，也可以是一个向量
function IttenBrightness=Brightness(value)
n=length(value);
value=value*100;
VeryDark=zeros(n,1);
Dark=zeros(n,1);
Middle=zeros(n,1);
Light=zeros(n,1);
VeryLight=zeros(n,1);

for i=1:n
    if value(i)<=21
        VeryDark(i)=1;
        Dark(i)=0;
        Middle(i)=0;
        Light(i)=0;
        VeryLight(i)=0;
    elseif value(i)<=39
        VeryDark(i)=(39-value(i))/18;
        Dark(i)=(value(i)-21)/18;
        Middle(i)=0;
        Light(i)=0;
        VeryLight(i)=0;
    elseif value(i)<=55
        VeryDark(i)=0;
        Dark(i)=(55-value(i))/16;
        Middle(i)=(value(i)-39)/16;
        Light(i)=0;
        VeryLight(i)=0;
    elseif value(i)<=68
        VeryDark(i)=0;
        Dark(i)=0;
        Middle(i)=(68-value(i))/13;
        Light(i)=(value(i)-55)/13;
        VeryLight(i)=0;
    elseif value(i)<=84
        VeryDark(i)=0;
        Dark(i)=0;
        Middle(i)=0;
        Light(i)=(84-value(i))/16;
        VeryLight(i)=(value(i)-68)/16;
    else
        VeryDark(i)=0;
        Dark(i)=0;
        Middle(i)=0;
        Light(i)=0;
        VeryLight(i)=1;
    end
end

IttenBrightness=cat(2,VeryDark,Dark,Middle,Light,VeryLight);

end