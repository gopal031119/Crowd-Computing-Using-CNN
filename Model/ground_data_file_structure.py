# file is saved as .py file because 'mat' is similar dictionary in python, nothing else

mat
{   
    '__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Fri Nov 18 20:05:04 2016', 
    '__version__': '1.0', 
    '__globals__': [], 
     
    """
        ### general structure  of image_info ###

        array1(
                [
                    [
                        array2(
                                [
                                    [
                                        (
                                            # array 2.1 is sparse matrix which is required to create density map of ground-truth
                                            array2.1( [ [ 524.17564209,  611.31179232],....,[  20.58878763,  594.48550124] ] ),
                                            array2.2( [ [277] ] )
                                        )
                                    ]
                                ]
                            )
                    ]
                ]
            )
    """

    # extra explanation about 2d numpy array
        # let a = np.array( [ [1,2,3],[4,5,6],[7,8,9] ] ) 
        # a.shape = (3,3)
        # so 'a' will look like , 
        # a = [
        #        [1,2,3],
        #        [4,5,6],
        #        [7,8,9]
        #     ]
        # so, a[0] = [1,2,4], a[0][0]=a[0,0]=1, a[0][1]=a[0,1]=2......

    # Preprocessing.py lines 78 and 79
        # mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        # mat["image_info"][0,0][0,0][0] this line can be understand using above structure of image_info


    'image_info':
    array(  # array1 starts
            [
                [   # arra1[0,0]
                    array(  # array2 starts
                            [
                                [   # array1[0,0][0,0] or array2[0,0]
                                    (   # tuple
                                        array(  # array2.1 starts   , arra1[0,0][0,0][0]
                                                [
                                                    [ 524.17564209,  611.31179232],
                                                    [  20.58878763,  594.48550124],
                                                                ...
                                                                ...
                                                                ...
                                                    [ 109.35965995,  139.72929032]
                                                ]
                                            ),   # array2.1 ends

                                        array(    # array2.2 starts   , arra1[0,0][0,0][1]
                                                [
                                                    [277]
                                                ]
                                            , dtype=uint16) # array2.2 ends
                                    )
                                ]
                            ]
                        , dtype=[('location', 'O'), ('number', 'O')])   # array2 ends
                ]
            ]
        , dtype=object)  #array1 ends

}