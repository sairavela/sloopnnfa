function data = customReadFcn(filename)
    	newImg = load(filename);
    	data = newImg.icolor;

