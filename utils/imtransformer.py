import cv2


class ImageTransforming:

    def __init__(self, image):
        self.image = image
        self.x_shape = image.shape[1]
        self.y_shape = image.shape[0]
    
    def change_color(self, color):
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, color)
        return self
    
    def fliping_image(self, flip_axis):
        """Flip image, where:
           0 - vertical flip,
           1 - horizontal flip.
        """
        self.image = cv2.flip(self.image, flip_axis)
        return self
    
    def scaling_image(self, image_scale_percent):
        """Scales the image vertically and horizontally 
        by a specified percentage."""
        width = int(self.x_shape * image_scale_percent / 100)
        height = int(self.y_shape * image_scale_percent / 100)
        dim = (width, height)
        self.image = cv2.resize(self.image, dim)
        return self.image

    def put_text_into_image(
        self,
        text_message,
        text_location=(50, 50),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1,
        font_color=(255, 0, 0)
    ):
        self.image = cv2.putText(
            self.image, 
            text_message, 
            text_location, 
            font, 
            font_scale, 
            font_color
        )
        return self