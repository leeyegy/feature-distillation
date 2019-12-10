for attack in  PGD  Momentum
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg.py --attack_method $attack --epsilon $epsilon
	done
done

for attack in FGSM STA
	do
		python jpeg.py --attack_method $attack --epsilon 0.00784
	done

