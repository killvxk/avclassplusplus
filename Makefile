demo:
	make tokens aliases lb fam pup prop clean

tokens:
	@echo -e '\e[32m Generic Token Detection (Optional) \e[m'
	head -n1 data/malheurReference_lb.json 
	head -n5 data/malheurReference_gt.tsv 
	python avclass_generic_detect.py -lb data/malheurReference_lb.json -gt data/malheurReference_gt.tsv -tgen 10 > malheurReference.gen 
	head -n10 malheurReference.gen

aliases:
	@echo -e '\e[32m Alias Detection (Optional) \e[m'
	head -n1 data/malheurReference_lb.json 
	python avclass_alias_detect.py -lb data/malheurReference_lb.json -nalias 100 -talias 0.98 > malheurReference.aliases
	head -n5 malheurReference.aliases

lb:
	@echo -e '\e[32m Labeling \e[m'
	head -n1 data/malheurReference_lb.json 
	python avclass_labeler.py -lb data/malheurReference_lb.json -v > malheurReference.labels
	head -n5 malheurReference.labels

fam:
	@echo -e '\e[32m Family Ranking \e[m'
	head -n1 data/malheurReference_lb.json 
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -fam > malheurReference.labels
	head -n5 malheurReference_lb.families

pup:
	@echo -e '\e[32m PUP Classification \e[m'
	head -n1 data/malheurReference_lb.json 
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -pup > malheurReference.labels
	head -n5 malheurReference.labels

gt:
	@echo -e '\e[32m Ground Truth Evaluation \e[m'
	head -n1 data/malheurReference_lb.json 
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -gt data/malheurReference_gt.tsv -eval > data/malheurReference.labels

prop:
	@echo -e '\e[32m Label Propagation \e[m'
	@echo 'Prepare .labels file and sample files'
	@echo 'unzip -P infected samples.zip'
	@echo 'python avclass_propagator.py -sampledir samples -labels data/demo.labels'
	@echo 'diff -ur data/demo.labels data/demo_pr.labels || exit 0'

clean:
	rm -f malheurReference*
	rm -rf *.pyc __pycache__
	rm -rf samples
	rm -f data/demo_pr.labels


