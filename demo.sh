for attack in  FGSM  Momentum
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg.py --attack_method $attack --epsilon $epsilon | tee log/cifar10_$attack\_$epsilon.txt
	done
done

