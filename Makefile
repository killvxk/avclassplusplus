tests:
	python avclass_labeler.py -lb data/malheurReference_lb.json -v > malheurReference.labels
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -fam > malheurReference.labels
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -pup > malheurReference.labels
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -pup -fam > malheurReference.labels
	python avclass_labeler.py -lb data/malheurReference_lb.json -v -gt data/malheurReference_gt.tsv -eval > data/malheurReference.labels
	python avclass_generic_detect.py -lb data/malheurReference_lb.json -gt data/malheurReference_gt.tsv -tgen 10 > malheurReference.gen 
	python avclass_alias_detect.py -lb data/malheurReference_lb.json -nalias 100 -talias 0.98 > malheurReference.aliases
	python -m compileall -q .

clean:
	rm -f malheurReference*
	rm -rf *.pyc __pycache__
