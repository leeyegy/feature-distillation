for attack in  JSMA
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg.py --attack_method $attack --epsilon $epsilon | tee logs/CIFAR10_$attack\_$epsilon.txt
	done
done

python jpeg.py --attack_method CW --epsilon 0.0 | tee logs/CIFAR10_CW_0.0.txt

