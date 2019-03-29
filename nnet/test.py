import tensorflow as tf
import modules as model

input1 = tf.random.uniform((1,3,256,256))
    ### PTV Network Path ###
ptv_conv1 = tf.layers.conv2d(input1, filters=64, kernel_size=1, use_bias=False, name="init_conv1", data_format='channels_first')
ptv_conv2 = tf.layers.conv2d(ptv_conv1, filters=64, kernel_size=7, strides=2, padding='same', use_bias=False, name="init_conv2", data_format='channels_first')
ptv_block1 = model.block_layer(ptv_conv2, 64, 3, 2, True, "ptv_block1", data_format='channels_first')
ptv_block2 = model.block_layer(ptv_block1, 128, 4, 2, True, "ptv_block2", data_format='channels_first')
ptv_block3 = model.block_layer(ptv_block2, 256, 6, 2, True, "ptv_block3", data_format='channels_first')
ptv_block4 = model.block_layer(ptv_block3, 512, 3, 2, True, "ptv_block4", data_format='channels_first')
# print ("p1: ", ptv_block1)
# print ("p2: ", ptv_block2)
# print ("p3: ", ptv_block3)
# print ("p4: ", ptv_block4)
### OAR-C4 etwork Path 4#
oct_conv1 = tf.layers.conv2d(input1, filters=64, kernel_size=1, use_bias=False, data_format='channels_first')
oct_conv2 = tf.layers.conv2d(oct_conv1, filters=64, kernel_size=7, strides=2, padding='same', use_bias=False, data_format='channels_first')
oct_block1 = model.block_layer(oct_conv2, 64, 3, 2, True, "oct_block1", data_format='channels_first')
oct_block2 = model.block_layer(oct_block1, 128, 4, 2, True, "oct_block2", data_format='channels_first')
oct_block3 = model.block_layer(oct_block2, 256, 6, 2, True, "oct_block3", data_format='channels_first')
oct_block4 = model.block_layer(oct_block3, 512, 3, 2, True, "oct_block4", data_format='channels_first')

embed = tf.add(ptv_block4, oct_block4)

### Anti-ResNet Network Path ###
antires_block1 = model.block_layer(embed, 1024, 3, 2, True, "antires_block1", data_format='channels_first', transpose=True)
antires_block1 = ptv_block3 + oct_block3 + antires_block1 # skip-connection 1
antires_block2 = model.block_layer(antires_block1, 512, 6, 2, True, "antires_block2", data_format='channels_first', transpose=True)
antires_block2 = ptv_block2 + oct_block2 + antires_block2 # skip-connection 2
antires_block3 = model.block_layer(antires_block2, 256, 4, 2, True, "antires_block3", data_format='channels_first', transpose=True)
antires_block3 = antires_block3 + ptv_block1 + oct_block1 # skip-connection 3
antires_block4 = model.block_layer(antires_block3, 64, 3, 2, True, "antires_block4", data_format='channels_first', transpose=True)     
antires_block4 = antires_block4 + ptv_conv2 + oct_conv2 # skip-connection 4
antires_block5 = model.block_layer(antires_block4, 64, 1, 2, True, "antires_block5", data_format='channels_first', transpose=True)  
antires_endconv = tf.layers.conv2d(antires_block5, filters=3, kernel_size=1, use_bias=False, name="final_conv", data_format='channels_first') 

print (antires_endconv.shape)
with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(antires_endconv))
    writer.close()
