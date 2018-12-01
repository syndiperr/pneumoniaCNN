package pneumoniaCNN;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

public class Resize {

	private static void cropImages(int numberOfImages, int divisionAmount, String locationToFindImages, String destinationToSaveImage){

		String base = locationToFindImages;

		File file = new File(base);
		File[] files = file.listFiles();

		int imagesRead = 1;
		int heightSize = 0;
		int widthSize = 0;

		if(numberOfImages == -1) {
			numberOfImages = files.length;
		}

		int averageWidth = 0;
		int averageHeight = 0;
		while(imagesRead < numberOfImages) {
			try {
				BufferedImage img = ImageIO.read(files[imagesRead]);
				averageHeight += img.getHeight();
				averageWidth += img.getWidth();
				imagesRead++;
			} catch (IOException e) {
				System.out.println("error with image");
			}
		}
		averageHeight = averageHeight/numberOfImages;
		averageWidth = averageWidth/numberOfImages;

		imagesRead = 1;
		while(imagesRead < numberOfImages) {
			try {
				BufferedImage img = ImageIO.read(files[imagesRead]);
				int halfHeight = img.getHeight()/2;
				int halfWidth = img.getWidth()/2;
				heightSize = img.getHeight()-img.getHeight()/divisionAmount;
				widthSize = img.getWidth()-img.getWidth()/divisionAmount;
				BufferedImage newImage = new BufferedImage(widthSize, heightSize, BufferedImage.TYPE_INT_RGB);
				for(int x = 0; x<widthSize; x++) {
					for(int y = 0; y<heightSize; y++) {
						newImage.setRGB(x, y, img.getRGB((halfWidth-(widthSize/2))+x, (halfHeight-(heightSize/2))+y));
					}
				}
				newImage = resize(newImage, averageWidth, averageHeight);
				ImageIO.write(newImage, "jpeg", new File(destinationToSaveImage+files[imagesRead].getName()));
				imagesRead++;
			} catch (IOException e) {
				System.out.println("error with image");
			}
		}
	}

	private static BufferedImage resize(BufferedImage img, int height, int width) {
		Image tmp = img.getScaledInstance(width, height, Image.SCALE_SMOOTH);
		BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D g2d = resized.createGraphics();
		g2d.drawImage(tmp, 0, 0, null);
		g2d.dispose();
		return resized;
	}


	public static void main(String[] args) {
		int numberOfImagesToCrop = -1;//grab all images from 'locationToGrabImages'
		int divisionAmount = 4;

		String destinationToSaveImages = "/Users/jordanjones/Desktop/normal/";
		String locationToGrabImages = "/Users/jordanjones/Desktop/Projects/pneumoniaCNN/chest_xray/train/NORMAL/";
		System.out.print("Cropping images...");
		cropImages(numberOfImagesToCrop, divisionAmount, locationToGrabImages, destinationToSaveImages);
		System.out.println("finished");

		destinationToSaveImages = "/Users/jordanjones/Desktop/infected/";
		locationToGrabImages = "/Users/jordanjones/Desktop/Projects/pneumoniaCNN/chest_xray/train/PNEUMONIA/";
		System.out.print("Cropping images...");
		cropImages(numberOfImagesToCrop, divisionAmount, locationToGrabImages, destinationToSaveImages);
		System.out.println("finished");
	}

}


