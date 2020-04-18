for attack in  DeepFool CW
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg_tiny_imagenet.py --attack_method $attack --epsilon $epsilon | tee logs/tiny_imagenet_$attack\_$epsilon.txt
	done
done

