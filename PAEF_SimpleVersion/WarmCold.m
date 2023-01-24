%�ú���������ůɫ��membership function������ɫ����ů��
%����ֵhueΪ��ɫ��ɫ��,hueΪ0��1֮���ֵ
%hue������һ��ֵ��Ҳ������һ������
%���ֵΪ��ɫ������Ӧ����ů���ĳ̶�
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