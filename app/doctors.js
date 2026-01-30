// app/doctors.js
import { FontAwesome, MaterialIcons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import {
  Image,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';

// Local doctor images
import dr1Image from '../assets/images/dr1.jpg';
import dr2Image from '../assets/images/dr2.jpg';
import dr3Image from '../assets/images/dr3.jpg';
import dr4Image from '../assets/images/dr4.jpg';

export default function DoctorsScreen() {
  const router = useRouter();

  const handleGoBack = () => {
    router.back(); // safer than push('/dashboard')
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleGoBack} style={styles.backButton}>
          <MaterialIcons name="arrow-back-ios" size={24} color="#555" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Doctors</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        {/* Doctor 1 */}
        <DoctorCard
          image={dr2Image}
          name="Dr. Mikaela Sherry Lopez"
          title="Optometrist"
          experience="10 years experience"
          focus="Focus: specializes in vision therapy and non-surgical management of strabismus in children and adolescents (18 years and below)."
          hours="Mon - Sat / 9 AM - 10 PM"
        />

        {/* Doctor 2 */}
        <DoctorCard
          image={dr1Image}
          name="Dr. Maria Cherry Lopez"
          title="Optometrist"
          experience="20 years experience"
          focus="Focus: comprehensive management of strabismus in adults (19 years and above), including both non-surgical and surgical options."
          hours="Mon - Sat / 9 AM - 10 PM"
        />

        {/* Doctor 3 */}
        <DoctorCard
          image={dr3Image}
          name="Dr. Jhon Francis Labis"
          title="Optometrist"
          experience="8 years experience"
          focus="Focus: specializes in pediatric eye care and early detection of vision problems in children."
          hours="Mon - Fri / 8 AM - 5 PM"
        />

        {/* Doctor 4 */}
        <DoctorCard
          image={dr4Image}
          name="Dr. Art James Marcial"
          title="Optometrist"
          experience="12 years experience"
          focus="Focus: provides comprehensive eye exams and non-surgical vision correction solutions for adults."
          hours="Tue - Sat / 10 AM - 7 PM"
        />
      </ScrollView>
    </SafeAreaView>
  );
}

// Reusable Doctor Card component
function DoctorCard({ image, name, title, experience, focus, hours }) {
  return (
    <View style={styles.card}>
      <View style={styles.profileRow}>
        <Image source={image} style={styles.profilePic} />
        <View style={styles.infoBlock}>
          <View style={styles.experienceBadge}>
            <FontAwesome
              name="trophy"
              size={16}
              color="#1E3A8A"
              style={{ marginRight: 5 }}
            />
            <Text style={styles.experienceText}>{experience}</Text>
          </View>
          <Text style={styles.focusText}>{focus}</Text>
        </View>
      </View>
      <View style={styles.nameCard}>
        <Text style={styles.doctorName}>{name}</Text>
        <Text style={styles.doctorTitle}>{title}</Text>
      </View>
      <View style={styles.hoursContainer}>
        <MaterialIcons
          name="access-time"
          size={18}
          color="#fff"
          style={{ marginRight: 5 }}
        />
        <Text style={styles.hoursText}>{hours}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F0F8FF',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 20,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backButton: {
    padding: 5,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1E3A8A',
  },
  content: {
    padding: 20,
    alignItems: 'center',
  },
  card: {
    width: '100%',
    backgroundColor: '#D6ECFF',
    borderRadius: 20,
    padding: 20,
    elevation: 4,
    marginBottom: 20,
  },
  profileRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 20,
  },
  profilePic: {
    width: 100,
    height: 100,
    borderRadius: 50,
    marginRight: 15,
  },
  infoBlock: {
    flex: 1,
  },
  experienceBadge: {
    flexDirection: 'row',
    backgroundColor: '#FFD700',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
    alignItems: 'center',
    marginBottom: 10,
  },
  experienceText: {
    fontSize: 13,
    color: '#1E3A8A',
    fontWeight: '600',
  },
  focusText: {
    color: '#1E3A8A',
    fontSize: 14,
    fontWeight: '500',
  },
  nameCard: {
    backgroundColor: '#fff',
    borderRadius: 15,
    paddingVertical: 10,
    alignItems: 'center',
    marginTop: 10,
  },
  doctorName: {
    fontSize: 18,
    color: '#1E3A8A',
    fontWeight: 'bold',
  },
  doctorTitle: {
    fontSize: 14,
    color: '#333',
  },
  hoursContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#65A3D5',
    borderRadius: 20,
    paddingVertical: 8,
    marginTop: 20,
  },
  hoursText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
