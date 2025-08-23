import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useEffect, useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Pressable, Image } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Ionicons from '@expo/vector-icons/Ionicons';

export default function LoggingScreen() {
  const [image, setImage] = useState<string | null>(null);
  const router = useRouter();
  const { meal: paramMeal } = useLocalSearchParams();
  const [meal, setMeal] = useState(paramMeal ?? '');

  useEffect(() => {
    if (paramMeal) {
      console.log('Open from add meal: ', paramMeal);
      setMeal(paramMeal);
    }
  }, [paramMeal]);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });
    if (result.assets && result.assets.length > 0) {
      console.log('select photo');
      const uri = result.assets[0].uri;
      setImage(uri);
    } else {
      console.log('no image selected');
    }
  };

  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (result.assets && result.assets.length > 0) {
      console.log('take photo');
      const uri = result.assets[0].uri;
      setImage(uri);
    } else {
      console.log('no image taken');
    }
  };

  const confirmPhoto = () => {
    if (!image) return;
    router.push({
      pathname: '/(logging)/summary',
      params: { image: image, meal: meal },
    });
  };

  return (
    <ThemedSafeAreaView edges={['top']} className="px-4 py-8">
      <View className="flex-1 items-center gap-8 pt-16">
        {image ? (
          <Image source={{ uri: image }} className="h-80 w-80 rounded-3xl" />
        ) : (
          <View className="h-80 w-80 items-center justify-center rounded-3xl border-2 border-secondary-500 bg-background-0">
            <Text className="text-secondary-500">
              <Ionicons name="fast-food-outline" size={48} />
            </Text>
          </View>
        )}
        <View className="gap-4">
          <Pressable className="button-white w-80" onPress={pickImage}>
            <Text className="font-bodySemiBold text-secondary-500">
              Pick a photo from gallery
            </Text>
          </Pressable>

          <Pressable className="button-white w-80" onPress={takePhoto}>
            <Text className="font-bodySemiBold text-secondary-500">
              Take a photo
            </Text>
          </Pressable>
        </View>
      </View>
      {image && (
        <Pressable onPress={confirmPhoto} className="button-rounded w-full">
          <Text className="font-bodySemiBold text-background-0">
            Use this photo
          </Text>
        </Pressable>
      )}
    </ThemedSafeAreaView>
  );
}
