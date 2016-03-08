# README.md

ptb.{train,valid,test}.txt $B$r(B JSAI2016 $B$K;H$($k$h$&$K!$JX59E*$K0J2<$N%k!<%k(B
$B$K=>$C$FJQ7A$7$^$7$?!#(Bptb.{train,valid,test}.txt $BFb$N4q?t9TL\$NJ8(B(n mod 2
== 1, where n means line number for each sentence) $B$NJ8>O$rLd$$$NJ8(B Q, $B8eB3(B
$B$9$k6v?t9TL\$NJ8(B(n mod 2 == 0)$B$rEz$((B A $B$H$_$J$7$F(B

1. $B4q?t9TL\$NJ8Kv$K(B <cntnxt> $B$rA^F~$7!$2~9T%3!<%I$r:o=|!$(B
2. $BD>8e$N4q?t9TL\$HO"7k$5$;$k(B

$B0J>e$r$9$k(B sed script $B$r(B convert.sed $B$K=q$$$F!$0J2<$NJQ49$r<B;\(B

```bash
for f in ptb.*; do
    ff=jsai2016${f}
    sed -f convert.sed ${f} > ${ff}
done
```

1. $B%Y!<%9%i%$%s%b%G%k(B

jsai2016ptb.py:

$B%b%G%k$N9=@.(B:
$B%b%G%k$O(B4$BAX$N%K%e!<%i%k%M%C%H%o!<%/$K$J$C$F$$$^$9!#(B
$BBh(B1$BAX!'C18lKd9~$_AX(B 650$B%K%e!<%m%s(B
$BBh(B2$BAX!'(BLSTM 650$B%K%e!<%m%s(B
$BBh(B3$BAX!'(BLSTM 650$B%K%e!<%m%s(B
$BBh(B4$BAX!'%=%U%H%^%C%/%9AX(B 10000$B%K%e!<%m%s(B

$BJQ99$7$?%O%$%Q!<%Q%i%a!<%?$O0J2<$N$H$*$j(B:

# n_epoch = 39   # number of epochs
n_epoch = 10
$B$I$3$^$G3X=,$5$;$k$+!#5$$,C;$$$+$i>/$J$/$7$?!#(B

# batchsize = 20   # minibatch size
batchsize = 500
$B%_%K%P%C%A$N%5%$%:!#D94|$N7ONs$r3X=,$5$;$kI,MW$,$k$?$a!#(B25 $BG\$ND9$5$K$7$?!#(B

# bprop_len = 35   # length of truncated BPTT
bprop_len = 35
BPTT $B$N2a5n$X$N$5$+$N$\$j!#(B35 $B$@$H==J,$@$m$&$1$l$I!$3X=,$N9bB.2=$N$?$a(B 5 $B$KJQ99(B

# grad_clip = 5    # gradient norm threshold to clip
grad_clip = 1
$B8{G[%/%j%C%W$NCM(B Graves $B$K=>$C$F(B 1 $B$KJQ99(B

$B;HMQ$9$k%G!<%?%;%C%H!'(B
jsai2016ptb.train.txt
jsai2016ptb.test.txt
jsai2016ptb.valid.txt

$B%3%^%s%I%i%$%s(B:
python jsa2015ptb.py


2. $BBPOC%b%G%k(B

$B%U%!%$%kL>(B: jsai2016ptb_dialogue.py

$B>e5-$r!$(BQ, A $B$K(B 2 $BOC<T$KJ,$1!$$=$l$>$l$NH/OC$K(B <pad> $B$rKd$a$k$3$H$r9T$C$?!#(B
$BBPOC%b%G%k$r?^<($9$k$H2<?^$N$h$&$K$J$k!#?^$G$O>eCJ$N(B LSTM $B$,(B Q $B$9$J$o$A(B
$B<ALdJ8!$2<CJ$N(B LSTM $B$,(B A $B$9$J$o$ABP1~$9$k1~EzJ8$G$"$k!#(B

<sos>     ...     <eos>    <pad>     ...     <pad>    <sos>    ...
LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
 |        |        |        |        |        |        |        |
 |        |        |        |        |        |        |        |
 v        v        v        v        v        v        v        v
<pad>    ....     <pad>    <sos>    ...     <eos>    <pad>    ....
LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM

$B>e$N(B LSTM $B$NBh(B2$BAX$NJ8L.>pJs$,2<$N(B LSTM $B$NBh(B1$BAX$X$NF~NO$H$J$k!#(B

3. S2S$B%b%G%k(B

Sutskever $B$i$N%b%G%k$K=>$($P87L)$JBPOC%b%G%k$O!$(BLSTM $B$+$i(B LSTM $B$X$N(B
$BLp0u$,0l2s$@$1$G$9!#(B

<sos>     ...     <eos>    <pad>     ...     <pad>    <sos>    ...
LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM
                   |
                   |
                   v
<pad>    ....     <pad>    <sos>    ...     <eos>    <pad>    ....
LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM --> LSTM

$B$3$l$+$i$D$/$j$^$9!#%4%a%s%J%5%$!#(B

# jsai2016
