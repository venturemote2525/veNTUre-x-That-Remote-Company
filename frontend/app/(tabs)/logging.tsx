import { Text, ThemedSafeAreaView, View } from '@/components/Themed';
import { useEffect, useState, useRef } from 'react';
import * as ImagePicker from 'expo-image-picker';
import { Image, Animated, Easing, Dimensions, ScrollView } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useAuth } from '@/context/AuthContext';
import { uploadImage } from '@/utils/food/api';
import LoadingScreen from '@/components/LoadingScreen';
import uuid from 'react-native-uuid';
import { AnimatedPressable } from '@/components/AnimatedComponents';
import { Colors } from '@/constants/Colors';
import { FontAwesomeIcon } from '@fortawesome/react-native-fontawesome';
import {
  faCamera,
  faImages,
  faUtensils,
  faArrowRight,
} from '@fortawesome/free-solid-svg-icons';
import { LinearGradient } from 'expo-linear-gradient';

const { width } = Dimensions.get('window');

const getStringParam = (param: string | string[] | undefined): string => {
  if (Array.isArray(param)) {
    return param[0] || '';
  }
  return param || '';
};

export default function LoggingScreen() {
  const { profile } = useAuth();
  const [image, setImage] = useState<string | null>(null);
  const router = useRouter();
  const { meal: paramMeal } = useLocalSearchParams();
  const [meal, setMeal] = useState(getStringParam(paramMeal));
  const [loading, setLoading] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const id = uuid.v4();

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const mealValue = getStringParam(paramMeal);
    if (mealValue) {
      console.log('Open from add meal: ', mealValue);
      setMeal(mealValue);
    }

    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
          duration: 800,
          easing: Easing.ease,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 800,
          easing: Easing.ease,
          useNativeDriver: true,
        }),
      ]),
    ).start();
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

      Animated.spring(slideAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }).start();
    } else {
      console.log('no image selected');
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Sorry, we need camera permissions to make this work!');
      return;
    }

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

      Animated.spring(slideAnim, {
        toValue: 1,
        friction: 8,
        tension: 40,
        useNativeDriver: true,
      }).start();
    } else {
      console.log('no image taken');
    }
  };

  const confirmPhoto = async () => {
    if (!image || !profile) return;
    try {
      setLoading(true);
      setUploadLoading(true);
      const path = await uploadImage(id, profile.user_id, image);
      setUploadLoading(false);

      setAiLoading(true);
      setAiLoading(false);

      setLoading(false);
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

  const buttonSlide = slideAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [100, 0],
  });

  const buttonOpacity = slideAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 1],
  });

  const formattedMeal = meal
    ? `${meal.charAt(0).toUpperCase() + meal.slice(1).toLowerCase()} Meal`
    : '';

  return (
    <ThemedSafeAreaView edges={['top']} className="flex-1">
      <ScrollView
        contentContainerStyle={{
          flexGrow: 1,
          paddingHorizontal: 20,
          paddingVertical: 20,
          paddingBottom: 100,
        }}
        showsVerticalScrollIndicator={false}>
        <View className="flex-1 items-center justify-between">
          <View className="mb-6 w-full items-center">
            <Text className="mb-2 font-heading text-head2 text-secondary-500">
              Log Your Meal
            </Text>
            <Text className="text-center font-body text-body2 text-primary-300">
              {meal
                ? `Logging your ${meal.toLowerCase()} meal`
                : 'Capture your food to track nutrition'}
            </Text>
          </View>

          <View className="mb-8 items-center justify-center">
            {image ? (
              <View className="relative">
                <Image
                  source={{ uri: `data:image/jpeg;base64,${image}` }}
                  className="h-72 w-72 rounded-3xl shadow-lg"
                />
                <LinearGradient
                  colors={['transparent', 'rgba(0,0,0,0.3)']}
                  className="absolute bottom-0 left-0 right-0 h-20 rounded-b-3xl"
                />
                <View className="absolute bottom-4 left-4 right-4">
                  <Text className="text-center font-bodyBold text-background-0">
                    Ready to analyze!
                  </Text>
                </View>
              </View>
            ) : (
              <View className="items-center justify-center">
                <Animated.View
                  style={{
                    transform: [{ scale: pulseAnim }],
                    opacity: pulseAnim.interpolate({
                      inputRange: [1, 1.1],
                      outputRange: [0.7, 1],
                    }),
                  }}
                  className="mb-4 h-60 w-60 items-center justify-center rounded-3xl border-2 border-dashed border-secondary-300 bg-secondary-100/50">
                  <Ionicons
                    name="fast-food-outline"
                    size={48}
                    color={Colors.light.colors.secondary[400]}
                  />
                </Animated.View>
                <Text className="text-center font-body text-body2 text-primary-300">
                  Take a photo or choose from gallery
                </Text>
              </View>
            )}
          </View>
        </View>

        <View className="mb-6 w-full gap-3">
          <AnimatedPressable
            className="flex-row items-center justify-center gap-2 rounded-xl bg-secondary-500 px-4 py-3"
            onPress={takePhoto}
            scaleAmount={0.95}>
            <View className="flex-row items-center justify-center">
              <FontAwesomeIcon icon={faCamera} size={16} color="white" />
              <Text className="ml-2 font-bodySemiBold text-body2 text-background-0">
                Take Photo
              </Text>
            </View>
          </AnimatedPressable>

          <AnimatedPressable
            className="flex-row items-center justify-center gap-2 rounded-xl border border-secondary-200 bg-background-0 px-4 py-3"
            onPress={pickImage}
            scaleAmount={0.95}>
            <View className="flex-row items-center justify-center">
              <FontAwesomeIcon
                icon={faImages}
                size={16}
                color={Colors.light.colors.secondary[500]}
              />
              <Text className="ml-2 font-bodySemiBold text-body2 text-secondary-500">
                Choose from Gallery
              </Text>
            </View>
          </AnimatedPressable>
        </View>

        {meal && (
          <View className="mb-8 flex-row items-center justify-center gap-2 rounded-full bg-secondary-100 px-4 py-2">
            <FontAwesomeIcon
              icon={faUtensils}
              size={14}
              color={Colors.light.colors.secondary[500]}
            />
            <Text className="font-bodySemiBold text-body3 text-secondary-500">
              {formattedMeal}
            </Text>
          </View>
        )}
      </ScrollView>

      <Animated.View
        style={{
          transform: [{ translateY: buttonSlide }],
          opacity: buttonOpacity,
          position: 'absolute',
          bottom: 30,
          left: 20,
          right: 20,
          zIndex: 10,
        }}>
        <AnimatedPressable
          onPress={confirmPhoto}
          className="flex-row items-center justify-center gap-2 rounded-full bg-success-500 px-6 py-4 shadow-lg"
          scaleAmount={0.95}>
          <View className="flex-row items-center justify-center">
            <Text className="font-bodyBold text-body1 text-background-0">
              Analyze Meal
            </Text>
            <FontAwesomeIcon
              icon={faArrowRight}
              size={16}
              color="white"
              style={{ marginLeft: 8 }}
            />
          </View>
        </AnimatedPressable>
      </Animated.View>
    </ThemedSafeAreaView>
  );
}
