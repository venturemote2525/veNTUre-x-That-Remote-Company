import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useEffect, useState } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Pressable, Image } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useAuth } from '@/context/AuthContext';
import { uploadImage } from '@/utils/food/api';
import LoadingScreen from '@/components/LoadingScreen';
import uuid from 'react-native-uuid';

export default function LoggingScreen() {
  const { profile } = useAuth();
  const [image, setImage] = useState<string | null>(null);
  const router = useRouter();
  const { meal: paramMeal } = useLocalSearchParams();
  const [meal, setMeal] = useState(paramMeal ?? '');
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const id = uuid.v4();

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
      base64: true,
    });
    if (result.assets && result.assets.length > 0) {
      console.log('select photo');
      const base64Data = result.assets[0].base64!;
      setImage(base64Data);
    } else {
      console.log('no image selected');
    }
  };

  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
      base64: true,
    });

    if (result.assets && result.assets.length > 0) {
      console.log('take photo');
      const base64Data = result.assets[0].base64!;
      setImage(base64Data);
    } else {
      console.log('no image taken');
    }
  };

  const confirmPhoto = async () => {
    if (!image || !profile) return;
    try {
      setLoading(true);
      // Upload to database
      setUploadLoading(true);
      const path = await uploadImage(id, profile.user_id, image);
      setUploadLoading(false);

      // TODO: Send to AI
      setAiLoading(true);
      setAiLoading(false);

      setLoading(false);
      // Go to summary screen
      router.push({
        pathname: '/(logging)/summary',
        params: { mealId: id, meal: meal, type: 'log' },
      });
    } catch (error) {
      console.log('Upload meal error: ', error);
    }
  };

  if (loading) {
    return (
      <ThemedSafeAreaView>
        <LoadingScreen
          text={
            uploadLoading
              ? 'Uploading photo'
              : aiLoading
                ? 'Analysing photo'
                : 'Loading'
          }
        />
      </ThemedSafeAreaView>
    );
  }

  return (
    <ThemedSafeAreaView edges={['top']} className="px-4 py-8">
      <View className="flex-1 items-center gap-8 pt-16">
        {image ? (
          <Image
            source={{ uri: `data:image/jpeg;base64,${image}` }}
            className="h-80 w-80 rounded-3xl"
          />
        ) : (
          <View className="h-80 w-80 items-center justify-center rounded-3xl border-2 border-secondary-500 bg-background-0">
            <Text className="text-secondary-500">
              <Ionicons name="fast-food-outline" size={48} />
            </Text>
          </View>
        )}
        <View className="gap-4">
          <Pressable className="button-white w-80" onPress={pickImage}>
            <Text className="font-bodySemiBold text-body2 text-secondary-500">
              Pick a photo from gallery
            </Text>
          </Pressable>

          <Pressable className="button-white w-80" onPress={takePhoto}>
            <Text className="font-bodySemiBold text-body2 text-secondary-500">
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
