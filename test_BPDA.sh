for iter in 15
do 
	for epsilon in 0.03137 0.06275
	do
		python test_BPDA.py --max_iterations $iter --epsilon $epsilon | tee logs/BPDA-$epsilon\-$iter.txt
	done
done
