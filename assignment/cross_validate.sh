#!/bin/sh
export PATH=/home/alice/deeplearning/bin:$PATH
rm -f validation.tsv

for id in NA06984 NA06985 NA06986 NA06989 NA06994 NA07000 NA07037 NA07048 NA07051 NA07056 NA07347 NA07357 NA10847 NA10851 \
  NA11829 NA11830 NA11831 NA11832 NA11840 NA11843 NA11881 NA11892 NA11893 NA11894 NA11918 NA11919 NA11920 NA11930 NA11931 \
  NA11932 NA11933 NA11992 NA11994 NA11995 NA12003 NA12004 NA12005 NA12006 NA12043 NA12044 NA12045 NA12046 NA12058 NA12144 \
  NA12154 NA12155 NA12156 NA12234 NA12249 NA12272 NA12273 NA12275 NA12282 NA12283 NA12286 NA12287 NA12340 NA12341 NA12342 \
  NA12347 NA12348 NA12383 NA12399 NA12400 NA12413 NA12414 NA12489 NA12546 NA12716 NA12717 NA12718 NA12748 NA12749 NA12750 \
  NA12751 NA12760 NA12761 NA12762 NA12763 NA12775 NA12776 NA12777 NA12778 NA12812 NA12813 NA12814 NA12815 NA12827 NA12828 \
  NA12829 NA12830 NA12842 NA12843 NA12872 NA12873 NA12874 NA12878 NA12889 NA12890 NA18486 NA18488 NA18489 NA18498 NA18499 \
  NA18501 NA18502 NA18504 NA18505 NA18507 NA18508 NA18510 NA18511 NA18516 NA18517 NA18519 NA18520 NA18522 NA18523 NA18525 \
  NA18526 NA18528 NA18530 NA18531 NA18532 NA18533 NA18534 NA18535 NA18536 NA18537 NA18538 NA18539 NA18541 NA18542 NA18543 \
  NA18544 NA18545 NA18546 NA18547 NA18548 NA18549 NA18550 NA18552 NA18553 NA18555 NA18557 NA18558 NA18559 NA18560 NA18561 \
  NA18562 NA18563 NA18564 NA18565 NA18566 NA18567 NA18570 NA18571 NA18572 NA18573 NA18574 NA18577 NA18579 NA18582 NA18591 \
  NA18592 NA18593 NA18595 NA18596 NA18597 NA18599 NA18602 NA18603 NA18605 NA18606 NA18608 NA18609 NA18610 NA18611 NA18612 \
  NA18613 NA18614 NA18615 NA18616 NA18617 NA18618 NA18619 NA18620 NA18621 NA18622 NA18623 NA18624 NA18625 NA18626 NA18627 \
  NA18628 NA18629 NA18630 NA18631 NA18632 NA18633 NA18634 NA18635 NA18636 NA18637 NA18638 NA18639 NA18640 NA18641 NA18642 \
  NA18643 NA18644 NA18645 NA18646 NA18647 NA18648 NA18740 NA18745 NA18747 NA18748 NA18749 NA18757 NA18853 NA18856 NA18858 \
  NA18861 NA18864 NA18865 NA18867 NA18868 NA18870 NA18871 NA18873 NA18874 NA18876 NA18877 NA18878 NA18879 NA18881 NA18907 \
  NA18908 NA18909 NA18910 NA18912 NA18915 NA18916 NA18917 NA18923 NA18924 NA18933 NA18934 NA18939 NA18940 NA18941 NA18942 \
  NA18943 NA18944 NA18945 NA18946 NA18947 NA18948 NA18949 NA18950 NA18951 NA18952 NA18953 NA18954 NA18956 NA18957 NA18959 \
  NA18960 NA18961 NA18962 NA18963 NA18964 NA18965 NA18966 NA18967 NA18968 NA18969 NA18970 NA18971 NA18972 NA18973 NA18974 \
  NA18975 NA18976 NA18977 NA18978 NA18979 NA18980 NA18981 NA18982 NA18983 NA18984 NA18985 NA18986 NA18987 NA18988 NA18989 \
  NA18990 NA18991 NA18992 NA18993 NA18994 NA18995 NA18997 NA18998 NA18999 NA19000 NA19001 NA19002 NA19003 NA19004 NA19005 \
  NA19006 NA19007 NA19009 NA19010 NA19011 NA19012 NA19054 NA19055 NA19056 NA19057 NA19058 NA19059 NA19060 NA19062 NA19063 \
  NA19064 NA19065 NA19066 NA19067 NA19068 NA19070 NA19072 NA19074 NA19075 NA19076 NA19077 NA19078 NA19079 NA19080 NA19081 \
  NA19082 NA19083 NA19084 NA19085 NA19086 NA19087 NA19088 NA19089 NA19090 NA19091 NA19092 NA19093 NA19095 NA19096 NA19098 \
  NA19099 NA19102 NA19107 NA19108 NA19113 NA19114 NA19116 NA19117 NA19118 NA19119 NA19121 NA19129 NA19130 NA19131 NA19137 \
  NA19138 NA19141 NA19143 NA19144 NA19146 NA19147 NA19149 NA19152 NA19153 NA19159 NA19160 NA19171 NA19172 NA19175 NA19184 \
  NA19185 NA19189 NA19190 NA19197 NA19198 NA19200 NA19201 NA19204 NA19206 NA19207 NA19209 NA19210 NA19213 NA19214 NA19222 \
  NA19223 NA19225 NA19235 NA19236 NA19238 NA19239 NA19247 NA19248 NA19256 NA19257
do
  cross_validate.py -e $id
done

exit