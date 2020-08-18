for attack in none
do 
	for epsilon in 0.0
	do 
		python jpeg.py --attack_method $attack --epsilon $epsilon | tee logs/new_cifar10_$attack\_$epsilon.txt
	done
done

