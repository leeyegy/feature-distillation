for attack in  STA
do 
	for epsilon in 0.0
	do 
		python jpeg_imagenet.py --attack_method $attack --epsilon $epsilon | tee ./logs/Imagenet_$attack\_$epsilon.txt
	done
done
