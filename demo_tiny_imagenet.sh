<<<<<<< HEAD
for attack in  DeepFool CW
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg_tiny_imagenet.py --attack_method $attack --epsilon $epsilon | tee logs/tiny_imagenet_$attack\_$epsilon.txt
=======
for attack in  FGSM  Momentum PGD
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
		python jpeg_tiny_imagenet.py --attack_method $attack --epsilon $epsilon | tee log/tiny_imagenet_$attack\_$epsilon.txt
>>>>>>> 4f9a70bfec15a9e143ff83d1f657a2558121f534
	done
done

