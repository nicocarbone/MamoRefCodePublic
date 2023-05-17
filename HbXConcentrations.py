from tqdm import tqdm

def MPPL_homo(ua, ups, rho, n, d, m):
    '''Mean partial pathlength of photons in CW reflectance for a homogeneous slab (based on Contini I, 1997)
    #   INPUT:
    #   ua = absorption coefficient, mm**-1
    #   ups = reduced scattering coefficient, mm**-1
    #   rho = source-detector separation, mm
    #   n = refractive index, -
    #   d = slab's thickness, mm
    #   m = number of terms in the sum
    #   OUTPUT
    #   y = MPPL, mm
    #   rho = source-detector array, mm
    '''
    import numpy as np
    D = 1/(3*ups)
    z0 = 1/ups
    A_factor = 1.71148-5.27938*n+5.10161*n**2-0.66712*n**3+0.11902*n**4
    ze = 2*A_factor*D

    # rho = np.arange(rhoi,rhof,drho)

    temp = 0
    temp2 = temp
    for ii in range(-m, m):

        z3m = -2*ii*d-4*ii*ze-z0
        z4m = -2*ii*d-(4*ii-2)*ze+z0
        aa3 = (rho**2+z3m**2)/D
        aa4 = (rho**2+z4m**2)/D

        cacho13 = z3m*(rho**2+z3m**2)**(-0.5)
        cacho23 = np.exp(-(ua*aa3)**(0.5))
        cacho14 = z4m*(rho**2+z4m**2)**(-0.5)
        cacho24 = np.exp(-(ua*aa4)**(0.5))
        factor3 = cacho13*cacho23
        factor4 = cacho14*cacho24
        temp = factor3-factor4
        temp2 = temp+temp2

    partR = -1/(8*np.pi*D)*temp2

    ref = ref_homo(ua, ups, rho, n, d, m)
    # Ref = np.reshape(ref.T,np.shape(partR))
    # for i in range(np.shape(partR)[1]):
    # y[i] = partR[i]/ref
    y = np.transpose(partR)/ref
    return y


def ref_homo(ua, ups, rho, n, d, m):
    '''#CW reflectance for a homogeneous slab (Contini I, 1997)
    #   INPUT:
    #   ua = absorption coefficient, mm**-1
    #   ups = reduced scattering coefficient, mm**-1
    #   rho = source-detector separation, mm
    #   n = refractive index, -
    #   d = slab's thickness, mm
    #   m = number of terms in the sum
    #   OUTPUT
    #   y = CW reflectance, number of photons per mm**2
    #   rho = source-detector array, mm
    '''
    import numpy as np
    import scipy as sp
    D = 1/(3*ups)
    z0 = 1/ups
    A_factor = 1.71148-5.27938*n+5.10161*n**2-0.66712*n**3+0.11902*n**4
    ze = 2*A_factor*D

    temp = 0
    temp2 = temp
    for ii in range(-m, m):

        z3m = -2*ii*d-4*ii*ze-z0
        z4m = -2*ii*d-(4*ii-2)*ze+z0

        cacho13 = z3m*(rho**2+z3m**2)**(-3/2)
        cacho23 = 1+((ua*(rho**2+z3m**2)/D)**(1/2))
        cacho33 = np.exp(-(ua*(rho**2+z3m**2)/D)**(1/2))
        cacho14 = z4m*(rho**2+z4m**2)**(-3/2)
        cacho24 = 1+((ua*(rho**2+z4m**2)/D)**(1/2))
        cacho34 = np.exp(-(ua*(rho**2+z4m**2)/D)**(1/2))
        factor3 = cacho13*cacho23*cacho33
        factor4 = cacho14*cacho24*cacho34
        temp = factor3-factor4
        temp2 = temp+temp2

    y = -1/(4*np.pi)*temp2
    return y


def MamoRef_dHbX(im, lambdas, srcPos, cropSizeX=100, cropSizeY=100, ua=0.01, ups=1, spectraData1="hbo.dat",
                 spectraData2="hbr.dat", concentration1=1, concentration2=1):
    '''Esta función toma como entrada un "stack" de imágenes de X x Y x lambdas 
    y las longitudes de onda especificadas como un vector de 1xlambdas (vector fila)

    # Se deben tener en la misma carpeta los archivos .dat de coeficientes de
    # extinción en función de lambda.
    '''

    import numpy as np
    import scipy as sp
    d = im

    # Densidad óptica

    # Calculo la densidad óptica bajo la suposición de que d contiene mapas de intensidad relativa
    OD = -np.log(d)

    # Búsqueda de lambdas
    # así como está, me quedo con la primera y la tercera. Se puede hacer más versátil.
    # lambdas = lambdas[0], lambdas[2]
    # Tomo las absorciones de las especies HbO y HbR para estas longitudes de onda.
    E = getspectra(lambdas, spectraData1, spectraData2,
                   concentration1, concentration2)
    
    # Hay que controlar que esta matriz no esté traspuesta. (Ahora no lo está.)

    # Propiedades ópticas. Habría que manejar esto desde afuera.
    n = 1.33
    m = 50
    d = 60  # Espesor del slab
    sizes = np.shape(im)
    sizeX = int(sizes[0])
    sizeY = int(sizes[1])

    res = np.zeros(np.shape(im))

    OD[np.isnan(OD)] = 0  # Tuve que hacerlo...

    invE = np.linalg.inv(E)

    L = np.ones((cropSizeX, cropSizeY))

    print("Step 1/2...")
    for i in tqdm(range(int(cropSizeX))):  # Basta con recorrer un cuarto del mosaico
        for j in range(int(cropSizeY)):
            rho = np.sqrt(i ^ 2+j ^ 2)

            L[i, j] = MPPL_homo(ua, ups, rho, n, d, m)

    LL = np.zeros((sizeX, sizeY))

    cont = np.zeros((sizeX, sizeY))

    print("Step 2/2...")
    for iC in tqdm(srcPos[:, 1]):
        for jC in srcPos[:, 0]:
            iC = int(iC)
            jC = int(jC)
            # LL[int(i-cropSizeX/2):int(i+cropSizeX/2), int(j-cropSizeY/2):int(j+cropSizeY/2)] += L
            # cont[int(i-cropSizeX/2):int(i+cropSizeX/2), int(j - cropSizeY/2):int(j+cropSizeY/2)] += 1
            # TODO: more efficient?
            for i in np.arange(-int(cropSizeX/2), int(cropSizeX/2)):
                for j in np.arange(-int(cropSizeY/2), int(cropSizeY/2)):
                    if iC+i > 0 and iC+i < sizeX and jC+j > 0 and jC+j < sizeY:
                        LL[iC+i, jC+j] += L[i, j]
                        cont[iC+i, jC+j] += 1

    LL = LL/cont

    for i in (0, 1):
        OD[:, :, i] = OD[:, :, i]/LL
    OD = np.transpose(OD, (0, 2, 1))
    res = np.dot(invE, OD)
    res = np.transpose(res, (1, 2, 0))

    return res


def getspectra(lambdas, spectraData1="hbo.dat", spectraData2="hbr.dat", concentration1=1, concentration2=1):
    #
    #   # # Hemoglobin
    # # http://omlc.ogi.edu/spectra/hemoglobin/
    # # scott prahl
    #
    # # # Fat
    # # by R.L.P. van Veen and H.J.C.M. Sterenborg, A. Pifferi, A. Torricelli and R. Cubeddu
    # # http://omlc.ogi.edu/spectra/fat/
    #
    # # # Water
    # # D. J. Segelstein, "The complex refractive index of water," University of
    # # Missouri-Kansas City, (1981).
    import numpy as np
    chromo1 = np.loadtxt(spectraData1)
    chromo2 = np.loadtxt(spectraData2)
    
    chromo1[:,1] = chromo1[:,1]/concentration1
    chromo2[:,1] = chromo2[:,1]/concentration2
    
    out = np.zeros((2, 2))
    out[:, 0] = np.interp(lambdas, chromo1[:, 0],
                          chromo1[:, 1])       # HbO extinction
    out[:, 1] = np.interp(lambdas, chromo2[:, 0],
                          chromo2[:, 1])       # HbR extinction
    return out
