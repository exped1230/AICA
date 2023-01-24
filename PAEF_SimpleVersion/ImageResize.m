%该函数对输入图像进行大小调整操作，为预处理的第二步
function Resied=ImageResize(rgb)

[n1,n2]=size(rgb(:,:,1));
SelectedSize=200000;
Ratio=sqrt(SelectedSize/(n1*n2));

m1=round(n1*Ratio);
m2=round(n2*Ratio);
Dif=abs(m1*m2-SelectedSize);
p1=m1;p2=m2;
for i=m1-1:m1+1
    for j=m2-1:m2+1
        Dif1=abs(i*j-SelectedSize);
        if Dif1<Dif
            p1=i;
            p2=j;
            Dif=Dif1;
        end
    end
end

Resied=imresize(rgb,[p1 p2]);

end

