import vgg16_torch
import vgg19_torch
import vgg16_more_classes
import vgg19_more_classes

if __name__ == '__main__':
   #vgg16_torch.vgg16() #96%
   #vgg19_torch.vgg19() #97%
   vgg16_more_classes.vgg16()
   #vgg19_more_classes.vgg19() #86%