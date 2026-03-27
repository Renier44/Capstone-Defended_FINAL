import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    TextInput,
    Image,
    Alert,
    Platform,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { MaterialIcons, Feather, Ionicons } from '@expo/vector-icons'; // Added Feather and Ionicons for potential use
import { useRouter } from 'expo-router';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as SecureStore from 'expo-secure-store';
import Constants from 'expo-constants'; // <-- FIX: Import Constants

const API_BASE_URL = 'https://capstone-defended-final.onrender.com/api';

// =======================================================
// 1. GLOBAL CONSTANTS (Matching previous component styles)
// =======================================================
const BRAND_BLUE = "#0057B7"; 
const PRIMARY_ACTION_COLOR = "#FFD54F"; // Yellow accent
const BACKGROUND_COLOR = "#E8F7FF"; // Light blue background
const NEUTRAL_TEXT = "#333333"; 
const INFO_BORDER_COLOR = "#77CDE0"; // Light blue border/accent color
const CARD_BG = "#fff"; // Clean white background for cards


export default function EditProfile() {
    const router = useRouter();

    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [dateOfBirth, setDateOfBirth] = useState('');
    const [gender, setGender] = useState(''); // Stores 'male' or 'female'
    const [profileImage, setProfileImage] = useState(null);
    const [isSaving, setIsSaving] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [showDatePicker, setShowDatePicker] = useState(false);
    const [fullNameDisplay, setFullNameDisplay] = useState('Loading...');

    const loadLocalUserProfile = async () => {
        try {
            const localProfileStr = await SecureStore.getItemAsync('userProfile');
            if (!localProfileStr) {
                console.warn('No local profile found. Redirecting to login.');
                router.replace('/login');
                return;
            }

            const localProfile = JSON.parse(localProfileStr);
            let apiFirstName = localProfile.first_name || '';
            let apiLastName = localProfile.last_name || '';
            const userEmail = localProfile.email || '';
            const userGender = localProfile.gender || ''; 
            const userDOB = localProfile.date_of_birth || '';

            let first = apiFirstName;
            let last = apiLastName;

            if (first && !last && first.includes(' ')) {
                const parts = first.trim().split(' ');
                first = parts[0] || '';
                last = parts.slice(1).join(' ') || '';
            }

            setFirstName(first);
            setLastName(last);
            setEmail(userEmail);
            setGender(userGender);
            setDateOfBirth(userDOB);
            setFullNameDisplay(`${first} ${last}`.trim() || 'Guest User');
            setProfileImage(localProfile.profile_image || null);
        } catch (error) {
            console.error('Error loading user profile:', error);
            Alert.alert('Error', 'Could not load profile data. Please log in again.');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        loadLocalUserProfile();
    }, []);

    const onChangeDate = (event, selectedDate) => {
        if (Platform.OS === 'android') setShowDatePicker(false);
        if (selectedDate) {
            const formattedDate = selectedDate.toISOString().split('T')[0];
            setDateOfBirth(formattedDate);
            if (Platform.OS === 'ios' && event.type === 'set') {
                setShowDatePicker(false);
            }
        } else {
            if (Platform.OS === 'ios' && event.type === 'dismissed') {
                setShowDatePicker(false);
            }
        }
    };

    const handleUpdateProfile = async () => {
        if (isSaving) return;
        if (!firstName || !lastName || !email || !gender || !dateOfBirth) {
            Alert.alert('Missing Fields', 'Please complete all required fields.');
            return;
        }

        setIsSaving(true);
        const userToken = await SecureStore.getItemAsync('userToken');

        if (!userToken) {
            Alert.alert('Authentication Error', 'You are not logged in.');
            router.replace('/login');
            return;
        }

        try {
            const payload = {
                first_name: firstName,
                last_name: lastName,
                email: email,
                gender: gender, 
                date_of_birth: dateOfBirth,
            };

            const response = await fetch(`${API_BASE_URL}/update-profile/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Token ${userToken}`,
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorData = JSON.parse(errorText);
                    throw new Error(errorData.error || errorData.message || `Failed with status ${response.status}`);
                } catch {
                    throw new Error(errorText || `Failed with status ${response.status}`);
                }
            }

            const data = await response.json();

            const updatedProfile = {
                first_name: data.first_name || firstName,
                last_name: data.last_name || lastName,
                email: data.email || email,
                gender: data.gender || gender,
                date_of_birth: data.date_of_birth || dateOfBirth,
                profile_image: data.profile_image || profileImage, 
                name: `${firstName} ${lastName}`, 
            };

            await SecureStore.setItemAsync('userProfile', JSON.stringify(updatedProfile));
            await loadLocalUserProfile(); 

            Alert.alert('Success', 'Profile updated successfully!');
            router.back();
        } catch (error) {
            console.error('Update error:', error);
            Alert.alert('Error', `Failed to update profile: ${error.message || 'Please try again.'}`);
        } finally {
            setIsSaving(false);
        }
    };

    if (isLoading) {
        return (
            <View style={[styles.container, styles.loadingContainer]}>
                <ActivityIndicator size="large" color={BRAND_BLUE} />
                <Text style={styles.loadingText}>Loading profile...</Text>
            </View>
        );
    }

    return (
        <SafeAreaView style={styles.container}>
            {/* Header matching the fixedHeader style of Profile.js */}
            <View style={styles.fixedHeader}>
                <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
                    <Ionicons name="arrow-back" size={24} color={BRAND_BLUE} />
                </TouchableOpacity>
                <Text style={styles.pageTitleFixed}>Edit Profile</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                
                {/* Profile Image Section */}
                <View style={styles.imageSection}>
                    <View style={styles.profilePicWrapper}>
                        <Image
                            source={{
                                uri: profileImage
                                    ? profileImage
                                    : 'https://cdn-icons-png.flaticon.com/512/149/149071.png',
                            }}
                            style={styles.profilePic}
                        />
                        {/* Edit icon placeholder - functionality needs to be added elsewhere */}
                        <View style={styles.editIconWrapper}>
                            <Feather name="edit-3" size={18} color="#fff" />
                        </View>
                    </View>
                    <Text style={styles.userName}>{fullNameDisplay}</Text>
                </View>

                {/* Form Inputs (Wrapped in Input Card style) */}
                <View style={styles.inputCard}>
                    <View style={styles.inputGroup}>
                        <Text style={styles.labelText}>First Name</Text>
                        <TextInput style={styles.input} value={firstName} onChangeText={setFirstName} />
                    </View>

                    <View style={styles.inputGroup}>
                        <Text style={styles.labelText}>Last Name</Text>
                        <TextInput style={styles.input} value={lastName} onChangeText={setLastName} />
                    </View>

                    <View style={styles.inputGroup}>
                        <Text style={styles.labelText}>Email</Text>
                        <TextInput
                            style={styles.input}
                            value={email}
                            onChangeText={setEmail}
                            keyboardType="email-address"
                            autoCapitalize="none"
                            editable={false} // Email should usually not be editable on this screen
                        />
                    </View>

                    {/* Date of Birth */}
                    <View style={styles.inputGroup}>
                        <Text style={styles.labelText}>Date of Birth</Text>
                        <TouchableOpacity style={styles.input} onPress={() => setShowDatePicker(true)}>
                            <Text style={[styles.inputPlaceholderText, dateOfBirth && styles.inputText]}>
                                {dateOfBirth || 'Select Date'}
                            </Text>
                        </TouchableOpacity>
                        {showDatePicker && (
                            <DateTimePicker
                                value={dateOfBirth ? new Date(dateOfBirth) : new Date()}
                                mode="date"
                                display={Platform.OS === 'ios' ? 'spinner' : 'default'}
                                onChange={onChangeDate}
                                maximumDate={new Date()}
                            />
                        )}
                    </View>

                    {/* Gender */}
                    <View style={styles.inputGroup}>
                        <Text style={styles.labelText}>Gender</Text>
                        <View style={styles.genderContainer}>
                            {['Male', 'Female'].map((option) => (
                                <TouchableOpacity
                                    key={option}
                                    style={[
                                        styles.genderOption,
                                        gender === option.toLowerCase() && styles.genderOptionActive, 
                                    ]}
                                    onPress={() => setGender(option.toLowerCase())} 
                                >
                                    <Text
                                        style={[
                                            styles.genderText,
                                            gender === option.toLowerCase() && styles.genderTextActive,
                                        ]}
                                    >
                                        {option}
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>
                </View>

                {/* Update Button */}
                <View style={styles.actionContainer}>
                    <TouchableOpacity
                        style={[styles.buttonPrimary, isSaving && { opacity: 0.7 }]}
                        onPress={handleUpdateProfile}
                        disabled={isSaving}
                    >
                        <Text style={styles.buttonPrimaryText}>
                            {isSaving ? 'UPDATING...' : 'UPDATE PROFILE'}
                        </Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
}

// =======================================================
// 2. STYLES DEFINITIONS (PROFESSIONAL THEME)
// =======================================================

const CARD_ELEVATION = {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 3,
    elevation: 3,
};

const styles = StyleSheet.create({
    // --- Overall Layout ---
    container: {
        flex: 1,
        backgroundColor: BACKGROUND_COLOR,
    },
    loadingContainer: { 
        flex: 1, 
        justifyContent: 'center', 
        alignItems: 'center', 
        backgroundColor: BACKGROUND_COLOR 
    },
    loadingText: { 
        marginTop: 10, 
        color: BRAND_BLUE, 
        fontFamily: "VarelaRound-Regular" 
    },
    scrollContent: {
        padding: 20,
        paddingTop: 0, // Adjusted padding
    },

    // --- Fixed Header ---
    fixedHeader: {
        height: Platform.OS === 'android' ? Constants.statusBarHeight + 50 : 60, 
        width: '100%',
        paddingHorizontal: 20,
        paddingBottom: 5, 
        backgroundColor: 'transparent', 
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center', 
    },
    pageTitleFixed: {
        fontSize: 22, 
        fontFamily: "Montserrat-VariableFont_wght",
        fontWeight: "800",
        color: BRAND_BLUE, 
        textAlign: "center",
        flex: 1, 
        marginTop: 40,
    },
    backButton: {
        position: "absolute",
        top: Platform.OS === 'android' ? Constants.statusBarHeight + 5 : 5, // Moved up
        left: 20,
        zIndex: 10,
        padding: 10,
        backgroundColor: "#fff",
        borderRadius: 50, 
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 3.84,
        elevation: 5,
    },


    // --- Profile Image Section ---
    imageSection: {
        alignItems: 'center',
        marginBottom: 30,
        paddingTop: 15, 
    },
    profilePicWrapper: {
        position: 'relative',
        marginBottom: 15,
    },
    profilePic: {
        width: 120, 
        height: 120,
        borderRadius: 60,
        backgroundColor: BACKGROUND_COLOR,
        borderWidth: 4,
        borderColor: INFO_BORDER_COLOR, 
    },
    editIconWrapper: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        backgroundColor: PRIMARY_ACTION_COLOR,
        borderRadius: 20,
        padding: 8,
        borderWidth: 2,
        borderColor: CARD_BG,
        ...CARD_ELEVATION,
    },
    userName: {
        fontSize: 24,
        fontWeight: '900',
        color: NEUTRAL_TEXT,
        fontFamily: "Montserrat-VariableFont_wght",
    },

    // --- Input Card Container ---
    inputCard: {
        marginBottom: 20,
        padding: 15,
        backgroundColor: CARD_BG,
        borderRadius: 15,
        ...CARD_ELEVATION,
        // Match the left border style from Profile.js
        borderLeftWidth: 5,
        borderLeftColor: INFO_BORDER_COLOR,
    },
    
    // --- Input Field Styles ---
    inputGroup: {
        marginBottom: 15,
    },
    labelText: {
        fontSize: 14,
        fontWeight: '600',
        color: BRAND_BLUE,
        marginBottom: 5,
        fontFamily: "VarelaRound-Regular",
    },
    input: {
        height: 50,
        borderColor: INFO_BORDER_COLOR,
        borderWidth: 1,
        borderRadius: 10,
        paddingHorizontal: 15,
        fontSize: 16,
        color: NEUTRAL_TEXT,
        backgroundColor: '#F7FCFF', // Very light background for inputs
        justifyContent: 'center',
        fontFamily: "VarelaRound-Regular",
    },
    inputPlaceholderText: {
        color: '#888',
        fontFamily: "VarelaRound-Regular",
    },
    inputText: {
        color: NEUTRAL_TEXT,
        fontFamily: "VarelaRound-Regular",
    },

    // --- Gender Selector ---
    genderContainer: { 
        flexDirection: 'row', 
        justifyContent: 'space-between', 
        marginTop: 5 
    },
    genderOption: { 
        flex: 1, 
        height: 50, 
        borderRadius: 10, 
        borderWidth: 1, 
        borderColor: INFO_BORDER_COLOR, // Use accent color for border
        justifyContent: 'center', 
        alignItems: 'center', 
        marginHorizontal: 5,
        backgroundColor: CARD_BG,
        ...CARD_ELEVATION,
    },
    genderOptionActive: { 
        borderColor: BRAND_BLUE, 
        backgroundColor: BACKGROUND_COLOR, // Light blue background when active
        borderWidth: 2,
    },
    genderText: { 
        color: NEUTRAL_TEXT,
        fontFamily: "VarelaRound-Regular",
    },
    genderTextActive: { 
        color: BRAND_BLUE, 
        fontWeight: 'bold',
        fontFamily: "VarelaRound-Regular",
    },

    // --- Action Button (Primary) ---
    actionContainer: {
        marginTop: 10,
        width: "100%",
        alignItems: "center"
    },
    buttonPrimary: { 
        width: "80%", 
        flexDirection: 'row', 
        justifyContent: 'center', 
        alignItems: 'center', 
        padding: 16, 
        borderRadius: 12, 
        backgroundColor: PRIMARY_ACTION_COLOR, // Yellow action color
        ...CARD_ELEVATION,
    },
    buttonPrimaryText: { 
        fontSize: 18, // Slightly larger font size
        fontWeight: "800", 
        color: BRAND_BLUE, // Blue text on yellow button
        fontFamily: "Montserrat-VariableFont_wght" 
    }, 
});