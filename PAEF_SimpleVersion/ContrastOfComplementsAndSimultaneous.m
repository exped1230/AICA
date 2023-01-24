%¸Ãº¯Êý¼ÆËãContrast of Complements
function Contrast=ContrastOfComplementsAndSimultaneous(hue)

n=length(hue);
hue=hue*360;
HueCategory=zeros(12);
for i=1:n
    if hue(i)<30
        HueCategory(1)=HueCategory(1)+1;
    elseif hue(i)<60
        HueCategory(2)=HueCategory(2)+1;
    elseif hue(i)<90
        HueCategory(3)=HueCategory(3)+1;
    elseif hue(i)<120
        HueCategory(4)=HueCategory(4)+1;
    elseif hue(i)<150
        HueCategory(5)=HueCategory(5)+1;
    elseif hue(i)<180
        HueCategory(6)=HueCategory(6)+1;
    elseif hue(i)<210
        HueCategory(7)=HueCategory(7)+1;
    elseif hue(i)<240
        HueCategory(8)=HueCategory(8)+1;
    elseif hue(i)<270
        HueCategory(9)=HueCategory(9)+1;
    elseif hue(i)<300
        HueCategory(10)=HueCategory(10)+1;
    elseif hue(i)<330
        HueCategory(11)=HueCategory(11)+1;
    else
        HueCategory(12)=HueCategory(12)+1;
    end
end

ComplementsContrast=zeros(1,3);
HuePair=zeros(1,6);
k=1;
for i=1:6
    if HueCategory(i) && HueCategory(6+i)
       ComplementsContrast(1)=ComplementsContrast(1)+1; 
       HuePair(k)=min(HueCategory(i),HueCategory(6+i))/max(HueCategory(i),HueCategory(6+i));
       k=k+1;
    end
end
if k>1
    HuePair=HuePair(1:k-1);
end
ComplementsContrast(2)=mean(HuePair);
ComplementsContrast(3)=std(HuePair);
if ComplementsContrast(1)
    SimultaneousContrast=0;
else 
    SimultaneousContrast=1;
end
Contrast=cat(2,ComplementsContrast,SimultaneousContrast);
