import { useEffect, useState } from 'react';
import { View, Image, Pressable, ActivityIndicator, Alert } from 'react-native';
import { Text, ThemedSafeAreaView } from '@/components/Themed';
import Ionicons from '@expo/vector-icons/Ionicons';
import * as ImagePicker from 'expo-image-picker';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'expo-router';
import { supabase } from '@/lib/supabase';
import uuid from 'react-native-uuid';

export default function ProfileScreen() {
  const { profile, user, profileLoading, loading } = useAuth();
  const [photo, setPhoto] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const router = useRouter();

  const email = user?.email ?? undefined;
  const memberSinceIso = profile?.created_at ?? user?.created_at;
  const memberSince = memberSinceIso ? new Date(memberSinceIso).toLocaleDateString() : undefined;
  const isBusy = loading || profileLoading;

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!loading && !user) {
      router.replace('/login');
    }
  }, [user, loading]);

  // Load avatar initially from Supabase Storage
  useEffect(() => {
    const loadAvatar = async () => {
      if (profile?.avatar_url) {
        const { data: urlData } = supabase.storage
          .from('avatars')
          .getPublicUrl(profile.avatar_url);

        setPhoto(urlData.publicUrl);
      }
    };
    loadAvatar();
  }, [profile]);

  // Pick image from gallery or camera
  const pickImage = async () => {
    if (!user) return;

    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission required', 'Please allow access to your photo library to upload images.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
        base64: true,
      });

      if (!result.canceled && result.assets.length > 0) {
        const base64Data = result.assets[0].base64;
        if (base64Data) {
          await uploadPhoto(base64Data);
        }
      }
    } catch (err) {
      console.log('Error picking image: ', err);
      Alert.alert('Error', 'Failed to pick image. Please try again.');
    }
  };

  // Upload photo to Supabase Storage (instant update)
  const uploadPhoto = async (base64Data: string) => {
    if (!user) return;

    try {
      setUploading(true);
      const id = uuid.v4();
      const fileName = `avatars/${user.id}-${id}.png`;
      const dataUrl = `data:image/png;base64,${base64Data}`;
      const response = await fetch(dataUrl);
      const blob = await response.blob();

      const { error: uploadError } = await supabase.storage
        .from('avatars')
        .upload(fileName, blob, { contentType: 'image/png', upsert: true });

      if (uploadError) {
        console.log('Upload error:', uploadError);
        throw uploadError;
      }

      // Get public URL instantly
      const { data: urlData } = supabase.storage.from('avatars').getPublicUrl(fileName);
      const publicUrl = urlData.publicUrl;

      // Update profile table in DB
      const { error: dbError } = await supabase
        .from('user_profiles')
        .update({ avatar_url: fileName })
        .eq('user_id', user.id);

      if (dbError) {
        console.log('Database error:', dbError);
        throw dbError;
      }

      // Set local state to instantly display new photo
      setPhoto(publicUrl);

      Alert.alert('Success', 'Profile picture updated successfully!');
    } catch (err) {
      console.log('Error uploading photo:', err);
      Alert.alert('Error', 'Failed to upload photo. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  // Logout
  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.log('Error signing out:', error);
        Alert.alert('Error', 'Failed to log out. Please try again.');
      } else {
        router.replace('/login');
      }
    } catch (err) {
      console.log('Logout error:', err);
      Alert.alert('Error', 'An unexpected error occurred during logout.');
    }
  };

  if (!user) {
    return (
      <ThemedSafeAreaView>
        <View className="flex-1 items-center justify-center">
          <ActivityIndicator size="large" />
        </View>
      </ThemedSafeAreaView>
    );
  }

  return (
    <ThemedSafeAreaView>
      <View className="flex-1 px-6 py-10 bg-background-50">
        {/* Profile Header */}
        <View className="items-center mb-6">
          <Pressable
            onPress={pickImage}
            disabled={uploading}
            className="rounded-full overflow-hidden"
            style={{ width: 96, height: 96 }}
          >
            <Image
              source={{
                uri: photo
                  ? photo
                  : `https://ui-avatars.com/api/?name=${encodeURIComponent(profile?.username ?? email ?? 'User')}&background=random`,
              }}
              style={{ width: 96, height: 96, borderRadius: 48 }}
            />
            <View
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: 96,
                height: 96,
                justifyContent: 'center',
                alignItems: 'center',
                backgroundColor: 'rgba(0,0,0,0.2)',
                borderRadius: 48,
              }}
            >
              {uploading ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <Ionicons name="camera-outline" size={24} color="#fff" />
              )}
            </View>
          </Pressable>

          <Text className="text-xl font-bodyBold text-primary-600 mt-2">
            {profile?.username || 'Guest User'}
          </Text>
          <Text className="text-sm text-gray-500">{isBusy ? 'Loading...' : email ?? 'No email'}</Text>
        </View>

        {/* Info Section */}
        <View className="space-y-4 bg-white rounded-2xl shadow-md p-6 mb-10">
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Member since</Text>
            <Text className="font-bodyBold text-black">{isBusy ? 'Loading...' : memberSince ?? 'N/A'}</Text>
          </View>
          <View className="h-[1px] bg-gray-200" />
          <View className="flex-row justify-between">
            <Text className="text-gray-500 font-body">Status</Text>
            <Text className="font-bodyBold text-green-600">Active</Text>
          </View>
        </View>

        {/* Logout Button */}
        <Pressable
          onPress={handleLogout}
          className="bg-red-500 py-3 rounded-2xl shadow-md"
        >
          <Text className="font-bodyBold text-background-0 text-center">Log out</Text>
        </Pressable>
      </View>
    </ThemedSafeAreaView>
  );
}
