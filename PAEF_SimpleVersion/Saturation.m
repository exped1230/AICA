%�ú������ñ��Ͷȵ�membership function���㱥�ͳ̶�
%����ֵcolorΪ��ɫ�ı��Ͷ�,colorΪ0��1֮���ֵ
%color������һ��ֵ��Ҳ������һ������
%���ֵΪ�ñ��Ͷ�����Ӧ�ĵ͡��С��߱��Ͷȵĳ̶�
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